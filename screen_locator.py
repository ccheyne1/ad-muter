"""Locate a screen from the current frame, including a partially clipped TV.

No saved camera coordinates are used. A partial screen is represented by its
visible upper-right bezel corner and two measured edge directions, not by an
invented fourth corner. This is a geometric screen detector, not a semantic
classifier: equally plausible displays are deliberately reported as ambiguous.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class Screen:
    corner: np.ndarray
    across: np.ndarray
    down: np.ndarray
    span: float
    vertical_span: float
    score: float
    method: str
    outline: np.ndarray
    tracking_used: bool = False

    def full_quad(self, frame_shape):
        h,w=frame_shape[:2]
        if len(self.outline)!=4 or any(x<.008*w or x>.992*w or y<.008*h or y>.992*h for x,y in self.outline):
            return None
        # Use the measured edge basis, so moderate camera roll doesn't change corner order.
        basis=np.column_stack((self.across,self.down))
        coordinates=np.linalg.solve(basis,(self.outline-self.corner).T).T
        tr=int(np.argmax(coordinates[:,0]-coordinates[:,1]))
        tl=int(np.argmin(coordinates[:,0]+coordinates[:,1]))
        br=int(np.argmax(coordinates[:,0]+coordinates[:,1]))
        bl=int(np.argmin(coordinates[:,0]-coordinates[:,1]))
        if len({tl,tr,br,bl})!=4:
            return None
        return np.float32(self.outline[[tl,tr,br,bl]])

    def header_quad(self, width=.42, depth=.15):
        # Offsets are relative to measured screen edges, never camera pixels.
        origin = self.corner - self.across * self.span * width
        inset = self.span * .008
        origin += self.down * inset
        horizontal = self.across * self.span * (width-.008)
        vertical = self.down * self.span * depth
        return np.float32([origin, origin+horizontal, origin+horizontal+vertical, origin+vertical])

    def moved_from(self, previous, frame_shape):
        if previous is None:
            return False
        diagonal = math.hypot(*frame_shape[:2])
        # Partial-screen edge lengths/endpoints change with video contours.
        # Scale and the lower edge basis are trustworthy only for complete quads.
        complete=self.full_quad(frame_shape) is not None and previous.full_quad(frame_shape) is not None
        return bool(np.linalg.norm(self.corner-previous.corner) > .025*diagonal
                or float(self.across@previous.across) < math.cos(math.radians(7))
                or complete and (abs(math.log(self.span/previous.span)) > .10
                                 or float(self.down@previous.down) < math.cos(math.radians(7))))

    def as_dict(self):
        return {'corner': self.corner.round(1).tolist(), 'across': self.across.round(4).tolist(),
                'down': self.down.round(4).tolist(), 'span': round(self.span,1),
                'score': round(self.score,3), 'method': self.method,
                'outline': self.outline.round(1).tolist(), 'tracking_used': self.tracking_used}


def _cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])


def _tracked_screen(candidates, previous, frame_shape):
    """Resolve a tie only with one strong current measurement near a known bezel."""
    if previous is None or previous.score<.85:return None
    diagonal=math.hypot(*frame_shape[:2])
    matches=[candidate for candidate in candidates
             if candidate.score>=.85
             and np.linalg.norm(candidate.corner-previous.corner)<=.02*diagonal
             and candidate.across@previous.across>=math.cos(math.radians(5))
             and candidate.down@previous.down>=math.cos(math.radians(15))
             and not candidate.moved_from(previous,frame_shape)]
    return matches[0] if len(matches)==1 else None


def _line_intersection(p,u,q,v):
    determinant = _cross(u,v)
    if abs(determinant) < .25:
        return None
    return p + u * (_cross(q-p,v)/determinant)


def _refine(contour, corner, across, down, span):
    points = contour.reshape(-1,2).astype(np.float32)
    directions, origins = [], []
    for direction, low, high in ((across,-.48,-.045),(down,.045,.38)):
        offset = points-corner
        along = offset@direction
        perpendicular = np.abs(offset[:,0]*direction[1]-offset[:,1]*direction[0])
        selected = points[(along>low*span)&(along<high*span)&(perpendicular<.045*span)]
        if len(selected) < 8:
            origins.append(corner); directions.append(direction); continue
        vx,vy,x,y = cv2.fitLine(selected,cv2.DIST_HUBER,0,.01,.01).reshape(-1)
        fitted = np.array([vx,vy],dtype=float)
        if fitted@direction < 0:
            fitted = -fitted
        if fitted@direction < math.cos(math.radians(12)):
            origins.append(corner); directions.append(direction); continue
        origins.append(np.array([x,y],dtype=float)); directions.append(fitted)
    intersection = _line_intersection(origins[0],directions[0],origins[1],directions[1])
    if intersection is not None and np.linalg.norm(intersection-corner) < .065*span:
        return intersection,directions[0],directions[1]
    return corner,across,down


def _edge_candidates(gray, factor):
    """Pair current dark bezel edges without requiring a closed content contour."""
    h,w=gray.shape
    smooth=cv2.GaussianBlur(gray,(5,5),1.2)
    dark=cv2.dilate(np.uint8(smooth<75)*255,np.ones((5,5),np.uint8))
    edges=cv2.bitwise_and(cv2.Canny(smooth,15,45),dark)
    lines=cv2.HoughLinesP(edges,1,np.pi/720,40,minLineLength=max(60,w*.07),maxLineGap=25)
    if lines is None:return []
    segments=[]
    for raw in lines.reshape(-1,4):
        p=raw[:2].astype(float);q=raw[2:].astype(float)
        if q[0]<p[0]:p,q=q,p
        length=float(np.linalg.norm(q-p));direction=(q-p)/length
        # Both edges must really lie against a dark strip, not simply a bright wall line.
        samples=p[None,:]+np.linspace(.08,.92,32)[:,None]*(q-p)[None,:]
        normal=np.array([-direction[1],direction[0]])
        probes=samples[:,None,:]+np.arange(-3,4)[None,:,None]*normal
        xx=np.clip(probes[:,:,0].astype(int),0,w-1);yy=np.clip(probes[:,:,1].astype(int),0,h-1)
        support=float(np.mean(gray[yy,xx].min(axis=1)<65))
        if support<.80:continue
        segments.append((p,q,direction,length,support))
    segments=sorted(segments,key=lambda item:item[3],reverse=True)[:80]
    # Wide-angle distortion and low contrast break the bezel into shorter segments.
    # Join adjacent, nearly collinear measurements, never extend past their endpoints.
    merged=[]
    for segment in segments:
        p,q,u,length,support=segment
        for number,(a,b,v,old_length,old_support) in enumerate(merged):
            if abs(float(u@v))<math.cos(math.radians(6)):continue
            normal=np.array([-v[1],v[0]])
            if min(abs(float((p-a)@normal)),abs(float((q-a)@normal)))>8:continue
            coordinates=np.array([float((p-a)@v),float((q-a)@v)])
            if coordinates.min()>old_length+35 or coordinates.max()<-35:continue
            points=np.float32([a,b,p,q])
            vx,vy,x,y=cv2.fitLine(points,cv2.DIST_L2,0,.01,.01).reshape(-1)
            direction=np.array([vx,vy],float)
            if direction[0]<0:direction=-direction
            center=np.array([x,y],float);positions=(points-center)@direction
            start=center+direction*positions.min();end=center+direction*positions.max()
            merged[number]=(start,end,direction,float(np.linalg.norm(end-start)),min(support,old_support))
            break
        else:merged.append(segment)
    segments=merged
    tops=[s for s in segments if s[2][0]>.88 and s[3]>w*.14]
    sides=[]
    for p,q,u,length,support in segments:
        if abs(u[1])<.70 or length<h*.18:continue
        if u[1]<0:p,q,u=q,p,-u
        sides.append((p,q,u,length,support))
    candidates=[]
    for p,q,across,top_length,top_support in tops:
        for a,b,down,side_length,side_support in sides:
            if _cross(across,down)<.55:continue
            corner=_line_intersection(p,across,a,down)
            if corner is None or not (.012*w<corner[0]<.988*w and .012*h<corner[1]<.988*h):continue
            if np.linalg.norm(corner-q)>min(100,top_length*.35) or np.linalg.norm(corner-a)>min(30,side_length*.15):continue
            span=float(np.linalg.norm(corner-p));vertical=float(np.linalg.norm(b-corner))
            if span<w*.14 or vertical<h*.20:continue
            # Require the room-facing side to be lighter than the black rim.
            checks=[]
            for start,end,outward in ((corner-across*span*.7,corner-across*span*.08,-down),
                                       (corner+down*vertical*.08,corner+down*vertical*.7,across)):
                centers=start[None,:]+np.linspace(0,1,32)[:,None]*(end-start)[None,:]
                outside=centers+outward*8
                rim=[]
                for offset in (-3,-1,1,3):
                    probe=centers+outward*offset
                    rim.append(gray[np.clip(probe[:,1].astype(int),0,h-1),np.clip(probe[:,0].astype(int),0,w-1)])
                values=gray[np.clip(outside[:,1].astype(int),0,h-1),np.clip(outside[:,0].astype(int),0,w-1)].astype(float)
                checks.append(float(np.mean(values-np.min(rim,axis=0)>6)))
            if min(checks)<.45:continue
            # Do not fabricate a fourth corner for a partially visible display.
            outline=np.array([corner-across*span,corner,corner+down*vertical])
            score=.88+min(.05,span/w*.08)+min(.04,vertical/h*.05)+.02*min(checks)
            candidates.append(Screen(corner/factor,across,down,span/factor,vertical/factor,
                                     min(.98,score),'bezel_edges',outline/factor))
    return candidates


class ScreenLocator:
    def __init__(self, max_width=1280, min_area=.065, ambiguity_margin=.075):
        self.max_width = max_width
        self.min_area = min_area
        self.ambiguity_margin = ambiguity_margin

    def locate(self, frame, previous=None):
        h,w = frame.shape[:2]
        factor = min(1.,self.max_width/w)
        small = cv2.resize(frame,(round(w*factor),round(h*factor))) if factor<1 else frame
        sh,sw = small.shape[:2]
        gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        if float(np.percentile(gray,95)-np.percentile(gray,5)) < 12:
            return None,'frame too flat or dark'
        masks = []
        # A thin, dark bezel often remains visible through large content changes.
        for threshold in (35,50,70):
            dark = np.uint8(gray<threshold)*255
            masks.append(('bezel',cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
        # Fallback for bezel-free displays and well-separated luminous screens.
        masks.append(('luminous',np.uint8((hsv[:,:,1]>65)&(hsv[:,:,2]>55))*255))
        masks.append(('luminous',np.uint8(hsv[:,:,2]>170)*255))
        candidates = _edge_candidates(gray,factor)
        for method,mask in masks:
            if method=='luminous':
                mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:12]:
                hull = cv2.convexHull(contour)
                area = cv2.contourArea(hull)
                fraction = area/(sw*sh)
                if not self.min_area < fraction < .88:
                    continue
                fill = cv2.contourArea(contour)/max(area,1)
                approx = cv2.approxPolyDP(hull,.013*cv2.arcLength(hull,True),True).reshape(-1,2)
                if not 3<=len(approx)<=7:
                    continue
                closed = len(approx)==4 and all(.008*sw<x<.992*sw and .008*sh<y<.992*sh for x,y in approx)
                if method=='bezel' and fill>.32 and not closed:
                    continue
                for i,point in enumerate(approx):
                    corner = point.astype(float)
                    if not (.012*sw<corner[0]<.988*sw and .012*sh<corner[1]<.988*sh):
                        continue
                    neighbors = [approx[(i-1)%len(approx)].astype(float)-corner,
                                 approx[(i+1)%len(approx)].astype(float)-corner]
                    for left,bottom in (neighbors,neighbors[::-1]):
                        length,vertical = np.linalg.norm(left),np.linalg.norm(bottom)
                        if length<sw*.18 or vertical<sh*.18:
                            continue
                        across,down = -left/length,bottom/vertical
                        if across[0]<.42 or down[1]<.42 or _cross(across,down)<.35:
                            continue
                        if method=='bezel':
                            corner,across,down = _refine(contour,corner,across,down,length)
                        if not (0<corner[0]<sw and 0<corner[1]<sh):
                            continue
                        # A geometric rectangle must enclose visible content, not just a wall line.
                        polygon = np.float32([corner-across*length*.50+down*length*.04,
                                              corner-across*length*.03+down*length*.04,
                                              corner-across*length*.03+down*length*.28,
                                              corner-across*length*.50+down*length*.28])
                        inside = np.zeros_like(gray)
                        cv2.fillConvexPoly(inside,polygon.astype(np.int32),255)
                        pixels = gray[inside>0]
                        if len(pixels)<500 or float(np.percentile(pixels,98)-np.percentile(pixels,2))<12:
                            continue
                        score = (.69 if method=='bezel' else .54) + min(.16,length/sw*.18) + min(.10,vertical/sh*.12)
                        if method=='bezel':
                            score += .05*(1-min(1,fill/.32))
                            if closed: score += .04
                        score += .08*fraction
                        candidates.append(Screen(corner/factor,across,down,length/factor,vertical/factor,
                                                 min(score,.99),method,approx.astype(float)/factor))
        if not candidates:
            return None,'screen not found'
        for candidate in candidates:
            if candidate.method!='bezel_edges':continue
            matching=next((other for other in sorted(candidates,key=lambda s:s.score,reverse=True)
                           if other.method=='bezel' and other.score>=.85
                           and np.linalg.norm(other.corner-candidate.corner)<.03*math.hypot(h,w)),None)
            if matching is not None:
                # Contours measure more of the screen when they remain intact;
                # paired lines are principally a fallback for broken contours.
                candidate.score=min(candidate.score,matching.score-.005)
        candidates.sort(key=lambda s:s.score,reverse=True)
        unique = []
        for candidate in candidates:
            matching=next((other for other in unique
                           if np.linalg.norm(candidate.corner-other.corner)<max(.075*min(candidate.span,other.span),.02*math.hypot(h,w))
                           and candidate.across@other.across>.92 and candidate.down@other.down>.90),None)
            if matching is not None:
                # Keep the most reliable local directions, but retain the longer
                # visible extent measured by another contour of the same bezel.
                if matching.method=='bezel_edges' and candidate.score>=.85:
                    matching.span=max(matching.span,candidate.span)
                    matching.vertical_span=max(matching.vertical_span,candidate.vertical_span)
                continue
            unique.append(candidate)
        best = unique[0]
        # A measured complete display remains preferable to a local pair of its edges.
        complete=next((candidate for candidate in unique if candidate.full_quad(frame.shape) is not None
                       and candidate.score>=.85 and np.linalg.norm(candidate.corner-best.corner)<best.span*.08),None)
        if complete is not None:best=complete
        # Content panels can have their own dark rectangular border. Prefer an
        # enclosing screen when a smaller candidate sits well inside its edges.
        def nested(candidate):
            basis=np.column_stack((best.across,best.down))
            offset=np.linalg.solve(basis,candidate.corner-best.corner)
            if (candidate.span<best.span*.65 and -.94*best.span<offset[0]<-.06*best.span
                    and abs(offset[1])<best.span*.04):
                return True
            if not (candidate.span<best.span*.94 and -.94*best.span<offset[0]<-.04*best.span
                    and .04*best.span<offset[1]<best.vertical_span*.85):
                return False
            # Clipped polygon vertices lie on the camera boundary, which is not
            # a measured screen edge. Compare only genuinely visible vertices.
            h,w=frame.shape[:2]
            visible=np.array([.008*w<x<.992*w and .008*h<y<.992*h for x,y in candidate.outline])
            if not visible.any():return False
            points=np.linalg.solve(basis,(candidate.outline[visible]-best.corner).T).T
            return bool(np.all((points[:,0]>=-best.span*1.2)&(points[:,0]<=best.span*.03)
                               &(points[:,1]>=-best.span*.03)&(points[:,1]<=best.vertical_span*1.12)))
        competing=[candidate for candidate in unique if candidate is not best and not nested(candidate)
                   and not (best.score>=.85 and candidate.score<.85)
                   and not (np.linalg.norm(candidate.corner-best.corner)<max(.075*min(candidate.span,best.span),.02*math.hypot(h,w))
                            and candidate.across@best.across>.92 and candidate.down@best.down>.90)]
        if competing and best.score-competing[0].score<self.ambiguity_margin:
            tracked=_tracked_screen([best,*competing],previous,frame.shape)
            if tracked is not None:
                tracked.tracking_used=True
                return tracked,None
            return None,'multiple plausible screens'
        return best,None


def rectify_header(frame, screen, width=.42, depth=.15):
    quad = screen.header_quad(width,depth)
    across = np.linalg.norm(quad[1]-quad[0])
    down = np.linalg.norm(quad[3]-quad[0])
    output_width = 1000
    output_height = max(100,min(550,round(output_width*down/across)))
    target = np.float32([[0,0],[output_width-1,0],[output_width-1,output_height-1],[0,output_height-1]])
    matrix = cv2.getPerspectiveTransform(quad,target)
    header = cv2.warpPerspective(frame,matrix,(output_width,output_height))
    valid = cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),matrix,(output_width,output_height),flags=cv2.INTER_NEAREST)
    return header,valid,quad


@dataclass
class BadgeProposal:
    box: tuple
    quad: np.ndarray

    def rectify(self, header):
        # The marker's own four edges correct its local shear/foreshortening.
        center=self.quad.mean(axis=0)
        source=np.float32(center+(self.quad-center)*1.04)
        target=np.float32([[0,0],[199,0],[199,89],[0,89]])
        return cv2.warpPerspective(header,cv2.getPerspectiveTransform(source,target),(200,90))


def badge_proposals(header, valid, limit=8):
    """Find small, approximately rectangular label backgrounds at any header position."""
    gray = cv2.cvtColor(header,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(header,cv2.COLOR_BGR2HSV)
    masks = [np.uint8((hsv[:,:,1]<145)&(hsv[:,:,2]>105))*255]
    high = float(np.percentile(gray,98))
    thresholds = {int(np.percentile(gray,q)) for q in (65,80,90)}
    thresholds.update(int(high*ratio) for ratio in (.65,.75,.82,.90))
    for threshold in sorted(thresholds):
        masks.append(np.uint8(gray>max(70,threshold))*255)
    candidates = []
    hh,hw = gray.shape
    for mask in masks:
        mask[valid==0] = 0
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if not (22<=w<=hw*.34 and 5<=h<=hh*.42 and 1.15<=w/h<=14):
                continue
            if y>hh*.65 or x<2 or y<2 or x+w>hw-2 or y+h>hh-2:
                continue
            fill = cv2.contourArea(contour)/(w*h)
            if fill<.65 or np.mean(valid[y:y+h,x:x+w]>0)<.98:
                continue
            if np.percentile(gray[y:y+h,x:x+w],98)-np.percentile(gray[y:y+h,x:x+w],2)<18:
                continue
            # Panel solidity is more informative than distance to a saved pixel position.
            poly = cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
            rectangle = len(poly) in (4,5) and cv2.isContourConvex(poly)
            score = fill + .30*(1-y/hh) + .20*x/hw + (.25 if rectangle else 0)
            hull=cv2.convexHull(contour)
            quad=None
            for epsilon in (.02,.03,.045):
                points=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True).reshape(-1,2)
                if len(points)==4:
                    sums=points.sum(axis=1);differences=points[:,0]-points[:,1]
                    order=[int(sums.argmin()),int(differences.argmax()),int(sums.argmax()),int(differences.argmin())]
                    if len(set(order))==4:
                        quad=np.float32(points[order]);break
            if quad is None:
                quad=np.float32([[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]])
            candidates.append((score,BadgeProposal((x,y,w,h),quad)))
    candidates.sort(key=lambda item:item[0],reverse=True)
    result = []
    for score,proposal in candidates:
        x,y,w,h = proposal.box
        if any(abs(x-a)+abs(y-b)<max(w,c)*.22 and abs(w-c)<max(w,c)*.3
               for a,b,c,d in (p.box for p in result)):
            continue
        result.append(proposal)
        if len(result)==limit:
            break
    return result
