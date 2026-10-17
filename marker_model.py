"""Locate and compare complete badge glyphs, independent of saved camera coordinates."""
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np


@dataclass
class MarkerCandidate:
    box: tuple
    quad: np.ndarray
    patch: np.ndarray
    kind: str = 'card'


def _uint8_percentiles(image, quantiles):
    """Linear percentiles of byte pixels, using 256 bins instead of pixel sorting."""
    histogram=cv2.calcHist([image],[0],None,[256],[0,256]).ravel()
    cumulative=np.cumsum(histogram)
    positions=(image.size-1)*(np.asarray(quantiles,dtype=float)/100)
    lower=np.floor(positions)
    fraction=positions-lower
    low=np.searchsorted(cumulative,lower,side='right').astype(float)
    high=np.searchsorted(cumulative,np.ceil(positions),side='right').astype(float)
    # Match NumPy's linear interpolation, including its rounding near the upper end.
    delta=high-low
    return np.where(fraction<.5,low+delta*fraction,high-delta*(1-fraction))


def proposals(frame,max_width=2304,word_gap=7,deadline=None,status=None):
    """Find compact bright cards anywhere in the current image, including warm light."""
    def expired():
        if deadline is not None and time.monotonic()>=deadline:
            if status is not None:status['complete']=False
            return True
        return False
    if status is not None:status['complete']=True
    if expired():return []
    h,w=frame.shape[:2];scale=min(1.,max_width/w)
    small=cv2.resize(frame,(round(w*scale),round(h*scale))) if scale<1 else frame
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    smooth=cv2.GaussianBlur(gray,(3,3),.6)
    edges=cv2.Canny(smooth,8,25)
    masks=[cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))]
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    for threshold in (65,100,140,180,210):
        masks.append(np.uint8((gray>threshold)&(hsv[:,:,1]<155))*255)
    found=[]
    for mask in masks:
        if expired():return found
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if expired():return found
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (18<=bw<=small.shape[1]*.12 and 3<=bh<=small.shape[0]*.06 and 2.2<=bw/bh<=14):continue
            hull=cv2.convexHull(contour)
            rectangle=cv2.minAreaRect(hull)
            rectangle_area=max(1,rectangle[1][0]*rectangle[1][1])
            solidity=cv2.contourArea(hull)/rectangle_area
            if solidity<.75:continue
            polygon=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True).reshape(-1,2)
            if len(polygon)!=4:
                if solidity<.90:continue
                polygon=cv2.boxPoints(rectangle)
            # Sum/difference ordering fails for a thin card rotated just a few
            # degrees. Its two left and two right vertices remain distinct.
            by_x=polygon[np.argsort(polygon[:,0])]
            left=by_x[:2][np.argsort(by_x[:2,1])]
            right=by_x[2:][np.argsort(by_x[2:,1])]
            quad=np.float32([left[0],right[0],right[1],left[1]])/scale
            box=tuple(round(v/scale) for v in (x,y,bw,bh))
            if any(abs(box[0]-c.box[0])+abs(box[1]-c.box[1])<max(box[2],c.box[2])*.2
                   and abs(box[2]-c.box[2])<max(box[2],c.box[2])*.25 for c in found):continue
            # Rectify the card's own edges; never rely on a screen-position crop.
            target=np.float32([[0,0],[199,0],[199,79],[0,79]])
            patch=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(quad,target),(200,80))
            pg=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            interior=pg[5:-5,8:-8]
            low,median,high=_uint8_percentiles(interior,[5,50,90])
            if median<50 or high-low<25:continue
            found.append(MarkerCandidate(box,quad,patch))
    # When a pale badge blends into pale video, its black letters still create
    # local contrast. Group those strokes instead of requiring a visible card border.
    if expired():return found
    blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((7,19),np.uint8))
    words=cv2.morphologyEx(np.uint8(blackhat>14)*255,cv2.MORPH_CLOSE,np.ones((2,word_gap),np.uint8))
    contours,_=cv2.findContours(words,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    text=[]
    for contour in contours:
        if expired():return found
        x,y,bw,bh=cv2.boundingRect(contour)
        if not (12<=bw<=100 and 3<=bh<=26 and 1.6<=bw/bh<=10):continue
        mx=max(5,round(bw*.12));my=max(3,round(bh*.5))
        x0=max(0,x-mx);y0=max(0,y-my);x1=min(gray.shape[1],x+bw+mx);y1=min(gray.shape[0],y+bh+my)
        region=gray[y0:y1,x0:x1]
        low,median,high=_uint8_percentiles(region,[10,50,90])
        if median<65 or high-low<25:continue
        # Compact strong ink on a bright field ranks before texture in video.
        quality=float(np.percentile(blackhat[y:y+bh,x:x+bw],90))
        text.append((quality,(x0,y0,x1-x0,y1-y0)))
    for quality,(x,y,bw,bh) in sorted(text,reverse=True)[:60]:
        if expired():return found
        quad=np.float32([[x,y],[x+bw-1,y],[x+bw-1,y+bh-1],[x,y+bh-1]])/scale
        patch=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(quad,np.float32([[0,0],[199,0],[199,79],[0,79]])),(200,80))
        box=tuple(round(v/scale) for v in (x,y,bw,bh))
        # Different proposal types are kept: one may contain a cleaner complete word.
        found.append(MarkerCandidate(box,quad,patch,'text'))
    return found


def glyph(patch):
    """Extract all interior ink and retain word proportions; borders are excluded."""
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY) if patch.ndim==3 else patch
    background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((25,41),np.uint8))
    ink=cv2.subtract(background,gray)
    ink=cv2.normalize(ink,None,0,255,cv2.NORM_MINMAX)
    binary=cv2.threshold(ink,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,round(gray.shape[1]*.8)),np.uint8))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((round(gray.shape[0]*.8),1),np.uint8))
    binary=cv2.bitwise_and(binary,cv2.bitwise_not(cv2.bitwise_or(horizontal,vertical)))
    # The card has explicit margins around the word. Remove its outer rim before
    # labeling ink, so a shadow connected to that rim cannot erase a whole letter.
    border=max(3,round(min(gray.shape)*.075))
    binary[:border]=0;binary[-border:]=0;binary[:,:border]=0;binary[:,-border:]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary)
    mask=np.zeros_like(binary)
    for number,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<12 or h<5:continue
        if x<=2 or y<=2 or x+w>=gray.shape[1]-2 or y+h>=gray.shape[0]-2:continue
        if w<5 and (x<gray.shape[1]*.08 or x+w>gray.shape[1]*.92):continue
        if (h>gray.shape[0]*.60 and w<gray.shape[1]*.25
                and (x<gray.shape[1]*.08 or x+w>gray.shape[1]*.92)):continue
        if (w>gray.shape[1]*.60 and h<gray.shape[0]*.25
                and (y<gray.shape[0]*.1 or y+h>gray.shape[0]*.9)):continue
        mask[labels==number]=255
    points=cv2.findNonZero(mask)
    if points is None:return None
    x,y,w,h=cv2.boundingRect(points)
    if h<8 or not .8<=w/h<=4.5 or w>gray.shape[1]*.92:return None
    crop=mask[y:y+h,x:x+w]
    factor=min(84/w,52/h)
    resized=cv2.resize(crop,(max(1,round(w*factor)),max(1,round(h*factor))),interpolation=cv2.INTER_AREA)
    out=np.zeros((64,96),np.uint8)
    top=(64-resized.shape[0])//2;left=(96-resized.shape[1])//2
    out[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
    return out


def descriptor(image):
    # NumPy HOG keeps the feature definition identical across Pi OpenCV 4 and
    # the PC's reduced OpenCV build; no optional cv2.HOGDescriptor is required.
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=1)
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=1)
    magnitude,angle=cv2.cartToPolar(gx,gy,angleInDegrees=True)
    position=(angle%180)/20
    lower=np.floor(position).astype(int)%9;fraction=position-np.floor(position)
    yy,xx=np.mgrid[:64,:96];cell=(yy//8)*12+xx//8
    hist=np.zeros((96,9),np.float32)
    np.add.at(hist,(cell.ravel(),lower.ravel()),(magnitude*(1-fraction)).ravel())
    np.add.at(hist,(cell.ravel(),((lower+1)%9).ravel()),(magnitude*fraction).ravel())
    hist=hist.reshape(8,12,9)
    blocks=[]
    for y in range(7):
        for x in range(11):
            block=hist[y:y+2,x:x+2].reshape(-1)
            block=block/np.sqrt(float(block@block)+1e-6)
            block=np.minimum(block,.2);block/=np.sqrt(float(block@block)+1e-6)
            blocks.append(block)
    vector=np.concatenate(blocks)
    return vector/max(float(np.linalg.norm(vector)),1e-6)


def complete_word(ink):
    """Require d's tall right stem; open C/c shapes cannot complete the word."""
    points=cv2.findNonZero(np.uint8(ink>100))
    if points is None:return False
    x,y,w,h=cv2.boundingRect(points);word=ink[y:y+h,x:x+w]>100
    if w<12 or h<10:return False
    left=np.flatnonzero(word[:,:max(1,round(w*.48))].any(axis=1))
    right=np.flatnonzero(word[:,round(w*.65):].any(axis=1))
    stem=word[:,round(w*.88):].sum(axis=0)
    return bool(len(left) and len(right) and max(left[0],right[0])<=h*.35
                and len(stem) and stem.max()/h>=.75)


class MarkerModel:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            if int(data.get('feature_version',0))!=3:raise ValueError('Unsupported marker feature version; retrain the model')
            self.templates=data['templates'].astype(np.float32)
            self.negatives=data['negatives'].astype(np.float32)
            self.threshold=float(data['threshold'])
            self.margin=float(data['margin'])
        if (self.templates.ndim!=2 or self.templates.shape[1]!=2772 or not len(self.templates)
                or self.negatives.ndim!=2 or self.negatives.shape[1]!=2772 or not len(self.negatives)
                or not np.isfinite(self.templates).all() or not np.isfinite(self.negatives).all()
                or not .5<=self.threshold<=1 or not 0<self.margin<.5):raise ValueError('Invalid marker model')

    def _score(self,found,deadline=None,complete=True):
        result={'accepted':False,'uncertain':not complete,'score':0.,'margin':0.,
                'candidate':None,'budget_exhausted':not complete}
        if deadline is not None and time.monotonic()>=deadline:
            result.update(uncertain=True,budget_exhausted=True)
            return result
        # Small batches bound CPU work and allow a strong positive to finish early.
        for start in range(0,len(found),12):
            candidates=[];vectors=[]
            for candidate in found[start:start+12]:
                if deadline is not None and time.monotonic()>=deadline:
                    result.update(uncertain=True,budget_exhausted=True)
                    break
                ink=glyph(candidate.patch)
                if ink is None:continue
                candidates.append((candidate,ink));vectors.append(descriptor(ink))
            if vectors:
                vectors=np.array(vectors,np.float32)
                positive=(vectors@self.templates.T).max(axis=1)
                negative=(vectors@self.negatives.T).max(axis=1)
                separation=positive-negative
                whole=np.array([complete_word(ink) for candidate,ink in candidates])
                accepted=(positive>=self.threshold)&(separation>=self.margin)&whole
                result['uncertain'] |= bool(((positive>=self.threshold-.05)&(separation>=self.margin-.04)).any())
                index=int(np.argmax(np.where(accepted,positive,positive-1))) if accepted.any() else int(np.argmax(positive))
                if accepted.any() or positive[index]>result['score']:
                    result.update(accepted=bool(accepted.any()),score=round(float(positive[index]),4),
                                  margin=round(float(separation[index]),4),candidate=candidates[index][0])
                if result['accepted']:return result
            if result['budget_exhausted']:break
        if deadline is not None and time.monotonic()>=deadline:
            result.update(uncertain=True,budget_exhausted=True)
        return result

    def inspect(self,frame,screen=None,deadline=None):
        status={}
        found=proposals(frame,deadline=deadline,status=status)
        first=self._score(found,deadline=deadline,complete=status['complete'])
        if first['accepted']:return first
        if first['budget_exhausted'] or deadline is not None and time.monotonic()>=deadline:
            first.update(uncertain=True,budget_exhausted=True)
            return first
        # Preserve the native pixels for a faint 4-8 pixel glyph. The second
        # search window comes from this frame's screen geometry, not saved points.
        x0=y0=0;region=frame
        if screen is not None and screen.score>=.80:
            quad=screen.header_quad(width=.70,depth=.25)
            margin=screen.span*.03
            x0=max(0,int(quad[:,0].min()-margin));y0=max(0,int(quad[:,1].min()-margin))
            x1=min(frame.shape[1],int(quad[:,0].max()+margin));y1=min(frame.shape[0],int(quad[:,1].max()+margin))
            if x1>x0 and y1>y0:region=frame[y0:y1,x0:x1]
            else:x0=y0=0
        found=proposals(region,max_width=region.shape[1],word_gap=19,deadline=deadline,status=status)
        for candidate in found:
            x,y,w,h=candidate.box;candidate.box=(x+x0,y+y0,w,h)
            candidate.quad+=np.float32([x0,y0])
        second=self._score(found,deadline=deadline,complete=status['complete'])
        if second['accepted']:return second
        selected=first if first['score']>=second['score'] else second
        selected['uncertain']=first['uncertain'] or second['uncertain']
        selected['budget_exhausted']=first['budget_exhausted'] or second['budget_exhausted']
        return selected
