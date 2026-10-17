"""Current-frame screen localization and dynamically discovered ad-marker OCR."""
from pathlib import Path
import json
import re
import time
import cv2
import numpy as np
from screen_locator import ScreenLocator, rectify_header, badge_proposals


def is_ad_token(text):
    match=re.fullmatch(r'[^\w]*([A-Za-z]+)[^\w]*',text.strip())
    return bool(match and match[1].casefold()=='ad')


def _confused_marker(reading):
    # These are uncertainty cues only. They can never count as an Ad vote.
    return any(c>=45 and re.sub(r'[^a-z]','',t.casefold()) in {'ac','ag','aa','aci','au','ai'}
               for t,c in reading['tokens'])


def _trim_word(gray):
    """Remove excess panel whitespace while retaining every interior glyph."""
    h,w=gray.shape
    if np.median(gray)<100:return None
    mask=np.uint8(gray<float(np.median(gray))-35)*255
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    boxes=[]
    for x,y,cw,ch,area in stats[1:]:
        if area<4:continue
        if x<=1 or y<=1 or x+cw>=w-1 or y+ch>=h-1:
            # A spanning frame is background; any other clipped ink is ambiguous.
            if cw<w*.75 and ch<h*.75:return None
            continue
        boxes.append((x,y,x+cw,y+ch))
    if not boxes:return None
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
    x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
    if y1-y0<4 or x1-x0>w*.85:return None
    crop=gray[max(0,y0-2):min(h,y1+2),max(0,x0-2):min(w,x1+2)]
    target_width=round(72*crop.shape[1]/crop.shape[0])
    if target_width>450:return None
    resized=cv2.resize(crop,(max(16,target_width),72),interpolation=cv2.INTER_CUBIC)
    return cv2.normalize(resized,None,0,255,cv2.NORM_MINMAX)


class Detector:
    def __init__(self, profile=None):
        self.cv,self.np=cv2,np
        self.profile=profile or {}
        self.confidence=float(self.profile.get('min_confidence',55))
        self.budget=float(self.profile.get('ocr_budget_seconds',2.4))
        self.marker_budget=float(self.profile.get('marker_budget_seconds',2.0))
        if not .25<=self.marker_budget<=2.4:raise ValueError('Invalid marker processing budget')
        self.corners=self.profile.get('search_corners',['top_right'])
        choices={'top_right','top_left','bottom_right','bottom_left'}
        if not self.corners or len(set(self.corners))!=len(self.corners) or any(c not in choices for c in self.corners):
            raise ValueError('search_corners must be a nonempty list of distinct screen corner names')
        if not 0<self.confidence<=100 or not .25<=self.budget<=10:
            raise ValueError('Invalid confidence or OCR budget')
        self.locator=ScreenLocator()
        self.previous=None
        self.previous_at=None
        self.previous_marker=None
        self.marker_model=None
        if self.profile.get('marker_model'):
            from marker_model import MarkerModel
            self.marker_model=MarkerModel(Path(__file__).parent/self.profile['marker_model'])
        self._ocr_callable=None  # Tests can inject a deterministic OCR backend.

    def _ocr(self, image, psm, deadline):
        import pytesseract
        remaining=deadline-time.monotonic()
        if remaining<.08:
            raise TimeoutError('OCR budget exhausted')
        if self._ocr_callable is not None:
            return self._ocr_callable(image,psm)
        return pytesseract.image_to_data(image,config=f'--psm {psm}',
                                        output_type=pytesseract.Output.DICT,timeout=min(1.2,remaining))

    @staticmethod
    def _reading(data):
        tokens=[(t,float(c)) for t,c in zip(data['text'],data['conf']) if t.strip()]
        confidence=max((c for t,c in tokens if is_ad_token(t)),default=-1.)
        return {'tokens':tokens,'ad_confidence':confidence}

    def _confirm(self, patch, deadline, seed=None):
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY) if patch.ndim==3 else patch
        gray=cv2.resize(gray,(200,90),interpolation=cv2.INTER_CUBIC)
        gray=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        trimmed=_trim_word(gray)
        if trimmed is not None:
            binary=cv2.threshold(trimmed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            variants=[(trimmed,7),(trimmed,8),(binary,8),(gray,8)]
        else:
            variants=[(gray,7),(cv2.createCLAHE(2,(4,4)).apply(gray),7),(gray,8)]
        readings=[]
        if seed is not None:
            readings.append(seed)
        i=0
        while i<len(variants):
            variant,psm=variants[i];i+=1
            pad=cv2.copyMakeBorder(variant,12,12,12,12,cv2.BORDER_CONSTANT,value=int(np.median(variant)))
            reading=self._reading(self._ocr(pad,psm,deadline))
            reading['match']=reading['ad_confidence']>=self.confidence
            readings.append(reading)
            confidences=sorted((r['ad_confidence'] for r in readings),reverse=True)
            if len(confidences)>=2 and (sum(c>=self.confidence for c in confidences)>=2
                                       or (confidences[0]>=85 and confidences[1]>=45)):
                return True,readings
            # One border-contaminated reading must not veto an otherwise readable
            # badge. Continue with the complete trimmed word; never crop letters.
        uncertain=max(r['ad_confidence'] for r in readings)>=30 or any(_confused_marker(r) for r in readings)
        return (None if uncertain else False),readings

    def _headers(self, frame, screen):
        full=screen.full_quad(frame.shape)
        if full is None:
            header,valid,quad=rectify_header(frame,screen)
            headers=[('top_right',header,valid,quad)] if 'top_right' in self.corners else []
            return headers, [c for c in self.corners if c!='top_right']
        # Four measured corners support a real projective rectification of the whole TV.
        target=np.float32([[0,0],[1919,0],[1919,1079],[0,1079]])
        transform=cv2.getPerspectiveTransform(full,target)
        inverse=np.linalg.inv(transform)
        view=cv2.warpPerspective(frame,transform,(1920,1080))
        headers=[]
        for corner in self.corners:
            x=1114 if corner.endswith('right') else 0
            y=792 if corner.startswith('bottom') else 0
            header=view[y:y+288,x:x+806]
            rect=np.float32([[[x,y],[x+805,y],[x+805,y+287],[x,y+287]]])
            quad=cv2.perspectiveTransform(rect,inverse)[0]
            headers.append((corner,header,np.full(header.shape[:2],255,np.uint8),quad))
        return headers,[]

    def _search_text(self, gray, valid, corner, partial, deadline):
        """Vote on the same word location across scales; never fuzzy-match Ad."""
        hh,hw=gray.shape
        strip_height=int(hh*.4)
        origin=0 if corner.startswith('top') else hh-strip_height
        strip=gray[origin:origin+strip_height]
        scales=[('full',3.),('full',4.),('strip',3.),('strip',4.),
                ('strip',3.5),('strip',4.5),('binary',4.)] if partial else [('full',1.),('binary_full',1.)]
        readings=[];buckets=[];uncertain=False
        for kind,scale in scales:
            source=gray if kind in ('full','binary_full') else strip
            offset=0 if kind in ('full','binary_full') else origin
            if kind.startswith('binary'):
                source=cv2.threshold(source,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            image=cv2.resize(source,(hw,round(source.shape[0]*scale)),interpolation=cv2.INTER_CUBIC)
            data=self._ocr(image,11,deadline)
            reading=self._reading(data)
            readings.append({'corner':corner,'search':'sparse text','variant':kind,'stretch':scale,'passes':[reading]})
            for i,text in enumerate(data['text']):
                x=int(data['left'][i]);y=offset+int(data['top'][i]/scale)
                w=int(data['width'][i]);h=max(1,int(data['height'][i]/scale))
                edge_y=y if corner.startswith('top') else hh-y-h
                if not (0<=x<x+w<=hw and 0<=y<y+h<=hh and w<=hw*.22 and edge_y<=hh*.35):
                    continue
                if np.mean(valid[y:y+h,x:x+w]>0)<.98:
                    continue
                item={'tokens':[(text,float(data['conf'][i]))],'ad_confidence':float(data['conf'][i]) if is_ad_token(text) else -1}
                uncertain |= _confused_marker(item)
                if not is_ad_token(text) or float(data['conf'][i])<30:
                    continue
                uncertain=True
                center=np.array([x+w/2,y+h/2])
                bucket=next((b for b in buckets if abs(center[0]-b['center'][0])<max(w,b['box'][2])*.45
                             and abs(center[1]-b['center'][1])<max(h,b['box'][3])*.8),None)
                if bucket is None:
                    bucket={'center':center,'box':[x,y,w,h],'votes':[],'last_pass':-1};buckets.append(bucket)
                if bucket['last_pass']==len(readings):continue
                bucket['votes'].append(float(data['conf'][i]));bucket['last_pass']=len(readings)
                values=sorted(bucket['votes'],reverse=True)
                if len(values)>=2 and (sum(v>=self.confidence for v in values)>=2 or values[0]>=85 and values[1]>=45):
                    return True,readings,bucket['box']
            if len(readings)>=3 and not uncertain:
                # Extra distortion passes are only useful when there is a possible marker.
                break
        return None if uncertain else False,readings,None

    def inspect(self, frame, diagnostic=None):
        out=Path(diagnostic) if diagnostic else None
        if out:
            out.mkdir(parents=True,exist_ok=True)
            for name in ('badge.png','header.png','region.jpg','screen.jpg','result.json'):
                (out/name).unlink(missing_ok=True)
        started=time.monotonic()
        result=self._inspect(frame,out,started+self.budget)
        result['processing_seconds']=round(time.monotonic()-started,3)
        if out:
            (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

    def _inspect(self, frame, out, deadline):
        if frame is None or frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8 or min(frame.shape[:2])<64:
            return {'ad':None,'reason':'invalid frame'}
        now=time.monotonic()
        if self.marker_model is not None and self.previous_at is not None and 0<=now-self.previous_at<=5:
            screen,reason=self.locator.locate(frame,previous=self.previous)
        else:
            screen,reason=self.locator.locate(frame)
        if self.marker_model is not None:
            model=self.marker_model.inspect(frame,screen=screen,deadline=min(deadline,time.monotonic()+self.marker_budget))
            result={'ad':None,'marker_similarity':model['score'],'marker_margin':model['margin'],
                    'marker_source':'glyph_model','geometry_changed':False}
            if screen is not None:result['screen']=screen.as_dict()
            if model['accepted']:
                candidate=model['candidate'];box=candidate.box
                if self.previous_marker is not None:
                    old,kind=self.previous_marker
                    result['geometry_changed']=bool(np.hypot(box[0]+box[2]/2-old[0]-old[2]/2,
                                                             box[1]+box[3]/2-old[1]-old[3]/2)>.025*np.hypot(*frame.shape[:2])
                                                    or (kind==candidate.kind and abs(np.log(max(box[2],1)/max(old[2],1)))>.35))
                elif screen is not None:
                    result['geometry_changed']=screen.moved_from(self.previous,frame.shape)
                self.previous_marker=(box,candidate.kind)
                if screen is not None and screen.score>=.80:
                    self.previous=screen;self.previous_at=time.monotonic()
                if out:
                    annotated=frame.copy();x,y,w,h=box
                    cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,210,0),5)
                    cv2.imwrite(str(out/'screen.jpg'),annotated);cv2.imwrite(str(out/'badge.png'),candidate.patch)
                return {**result,'ad':True,'marker_frame_box':list(box),'reason':'complete Ad badge matched learned glyphs'}
            self.previous_marker=None
            if model['uncertain']:
                return {**result,'reason':'marker processing budget exhausted' if model.get('budget_exhausted') else 'badge glyphs are ambiguous'}
            if screen is not None and screen.score>=.80:
                # Hulu's light-card preset searches that cue across the whole frame.
                # Distinguish camera blur from intentionally blurred/dark video content.
                size=(min(1280,frame.shape[1]),round(frame.shape[0]*min(1,1280/frame.shape[1])))
                gray=cv2.cvtColor(cv2.resize(frame,size),cv2.COLOR_BGR2GRAY)
                focus=float(cv2.Laplacian(gray,cv2.CV_64F).var())
                result['frame_sharpness']=round(focus,2)
                result['geometry_changed']=screen.moved_from(self.previous,frame.shape)
                self.previous=screen
                self.previous_at=time.monotonic()
                if focus>=5:
                    return {**result,'ad':False,'reason':'visible screen; no matching Hulu badge'}
                return {**result,'reason':'camera image is too blurred to confirm badge absence'}
            # Preserve uncertainty if the display has gone out of view. Generic
            # profiles without a marker model retain the OCR pipeline below.
            # Missing geometry is uncertainty, not proof that the camera moved.
            # Retain the last credible corner to detect a bump when it reappears.
            return {**result,'reason':reason or 'screen geometry or resolution is insufficient'}
        if screen is None:
            self.previous=None
            return {'ad':None,'reason':reason,'blank_screen':reason=='frame too flat or dark',
                    'geometry_changed':True}
        moved=screen.moved_from(self.previous,frame.shape)
        self.previous=screen
        result={'ad':None,'screen':screen.as_dict(),'geometry_changed':moved}
        if screen.score<.80 or screen.span<360:
            return {**result,'reason':'screen geometry or resolution is insufficient'}
        headers,unavailable=self._headers(frame,screen)
        result['unavailable_corners']=unavailable
        if not headers:
            return {**result,'reason':'requested screen corner is not visible'}
        if out:
            annotated=frame.copy()
            full=screen.full_quad(frame.shape)
            if full is not None:
                cv2.polylines(annotated,[full.astype(np.int32)],True,(0,210,0),6)
            else:
                corner=tuple(np.int32(screen.corner))
                left=tuple(np.int32(screen.corner-screen.across*screen.span))
                bottom=tuple(np.int32(screen.corner+screen.down*screen.vertical_span))
                cv2.line(annotated,left,corner,(0,210,0),6)
                cv2.line(annotated,corner,bottom,(0,210,0),6)
            for _,_,_,quad in headers:
                cv2.polylines(annotated,[quad.astype(np.int32)],True,(0,0,255),6)
            cv2.imwrite(str(out/'screen.jpg'),annotated)
            cv2.imwrite(str(out/'region.jpg'),annotated)
            cv2.imwrite(str(out/'header.png'),headers[0][1])
        uncertain=bool(unavailable)
        all_readings=[]
        for corner,header,valid,quad in headers:
            gray=cv2.cvtColor(header,cv2.COLOR_BGR2GRAY)
            coverage=float(np.mean(valid>0))
            result['region_coverage']=round(coverage,3)
            if coverage<.98:
                uncertain=True
            pixels=gray[valid>0]
            if len(pixels)<1000:
                return {**result,'reason':'ad search region is outside the camera view'}
            contrast=float(np.percentile(pixels,95)-np.percentile(pixels,5))
            sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            result.update(contrast=round(contrast,1),sharpness=round(sharpness,1))
            proposals=badge_proposals(header,valid,limit=4)
            result['candidate_count']=len(proposals)
            # Compact labels are faster and cleaner than OCR over moving video.
            # A dark/low-contrast scene can still contain a readable bright badge.
            try:
                for proposal in proposals:
                    patch=proposal.rectify(header)
                    accepted,readings=self._confirm(patch,deadline)
                    all_readings.append({'corner':corner,'box':list(proposal.box),'passes':readings})
                    if accepted:
                        if out:cv2.imwrite(str(out/'badge.png'),patch)
                        return {**result,'ad':True,'marker_box':list(proposal.box),'marker_corner':corner,'readings':all_readings}
                    uncertain |= accepted is None
                    if deadline-time.monotonic()<.35:break
            except (TimeoutError,RuntimeError) as error:
                return {**result,'ad':None,'reason':'OCR timeout or budget exhausted','detail':str(error),'readings':all_readings}
            if pixels.mean()<18 or (contrast>=12 and sharpness<4):
                return {**result,'reason':'ad search region is too dark or blurred'}
            if contrast<12:
                continue
            try:
                accepted,readings,box=self._search_text(gray,valid,corner,screen.full_quad(frame.shape) is None,deadline)
                all_readings.extend(readings)
                if accepted:
                    x,y,w,h=box
                    patch=header[max(0,y-3):min(header.shape[0],y+h+3),max(0,x-8):min(header.shape[1],x+w+8)]
                    if out:cv2.imwrite(str(out/'badge.png'),patch)
                    return {**result,'ad':True,'marker_box':box,'marker_corner':corner,'readings':all_readings}
                uncertain |= accepted is None
            except (TimeoutError,RuntimeError) as error:
                return {**result,'ad':None,'reason':'OCR timeout or budget exhausted','detail':str(error),'readings':all_readings}
        return {**result,'ad':None if uncertain else False,
                'reason':'incomplete or uncertain marker search' if uncertain else 'no ad marker found',
                'readings':all_readings}
