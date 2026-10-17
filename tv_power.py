"""Camera evidence for a received Power toggle; never infer state from press count."""
import cv2
import numpy as np
from screen_locator import ScreenLocator


class PowerCheck:
    """Settle after Power, then require fresh sustained evidence. Unknown is explicit."""
    def __init__(self, timeout=45):
        self.timeout=timeout
        self.active=False
        self.pending=False
        self.last=None
        self.candidate=None
        self.since=0
        self.count=0

    def request(self, now):
        self.before=self.active
        self.started=now
        self.pending=True
        self.candidate=None
        self.count=0
        self.last=None

    def observe(self, evidence, now):
        if not self.pending:return None
        if now-self.started>=self.timeout:
            self.pending=False
            return 'unknown'
        # A previously running TV may show a shutdown animation before going dark.
        settle=6 if self.before else 2
        if now-self.started<settle:return None
        if self.last is not None and (now<self.last or now-self.last>3):
            self.candidate=None;self.count=0
        self.last=now
        if evidence not in ('on','off'):
            self.candidate=None;self.count=0;return None
        if evidence!=self.candidate:
            self.candidate=evidence;self.since=now;self.count=0
        self.count+=1
        duration=4 if evidence=='off' else (3 if self.before else 1)
        if self.count>=3 and now-self.since>=duration:
            self.active=evidence=='on'
            self.pending=False
            return evidence
        return None


class ScreenPowerObserver:
    """Track a visible lit display; darkness alone without a baseline is unknown.

    The cached polygon is only a recent power-check reference, never an OCR crop.
    A moved camera invalidates the off comparison; positive geometry is reacquired.
    """
    def __init__(self, locator=None):
        self.locator=locator if locator is not None else ScreenLocator()
        self.screen=None
        self.shape=None
        self.lit=None
        self.anchor=None
        self.check_anchor=None
        self.check_lit=None
        self.off_allowed=False

    @staticmethod
    def _patch(frame, screen):
        quad=screen.header_quad(width=.45,depth=.22)
        target=np.float32([[0,0],[239,0],[239,119],[0,119]])
        matrix=cv2.getPerspectiveTransform(quad,target)
        patch=cv2.warpPerspective(frame,matrix,(240,120))
        valid=cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),matrix,(240,120),flags=cv2.INTER_NEAREST)
        if np.mean(valid>0)<.98:return None
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        return {'mean':float(gray.mean()),'p90':float(np.percentile(gray,90)),
                'range':float(np.percentile(gray,95)-np.percentile(gray,5))}

    @staticmethod
    def _anchor(frame, screen):
        # Compare the room outside the last screen outline to catch camera movement.
        small=cv2.resize(frame,(160,90))
        mask=np.full((90,160),255,np.uint8)
        scale=np.array([160/frame.shape[1],90/frame.shape[0]])
        quad=screen.full_quad(frame.shape)
        if quad is not None:
            cv2.fillConvexPoly(mask,(quad*scale).astype(np.int32),0)
        else:
            # A clipped bezel hull omits much of the panel. Exclude the entire
            # inward sector of its two visible edges, including unseen corners,
            # so changing TV content cannot contaminate the *room* comparison.
            yy,xx=np.mgrid[:90,:160]
            points=np.column_stack((xx.ravel()/scale[0],yy.ravel()/scale[1]))
            coordinates=np.linalg.solve(np.column_stack((screen.across,screen.down)),(points-screen.corner).T).T
            inward=(coordinates[:,0]<screen.span*.04)&(coordinates[:,1]>-screen.span*.04)
            mask.reshape(-1)[inward]=0
        mask=cv2.erode(mask,np.ones((7,7),np.uint8))
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(float)
        return gray,mask>0

    def request(self, allow_off):
        self.off_allowed=bool(allow_off and self.screen is not None and self.lit is not None)
        self.check_anchor=self.anchor
        self.check_lit=dict(self.lit) if self.lit is not None else None

    def inspect(self, frame, pending=False):
        screen,reason=self.locator.locate(frame)
        if screen is not None and screen.score>=.80:
            stats=self._patch(frame,screen)
            if stats and stats['mean']>18 and stats['p90']>40 and stats['range']>15:
                if self.screen is not None and screen.moved_from(self.screen,frame.shape):
                    self.off_allowed=False
                self.screen=screen;self.shape=frame.shape
                self.lit=stats;self.anchor=self._anchor(frame,screen)
                return 'on',{'screen':screen.as_dict(),'brightness':stats}
        if not pending or not self.off_allowed or frame.shape!=self.shape:
            return 'unknown',{'reason':reason or 'no reliable lit-screen baseline'}
        if self.check_anchor is None:return 'unknown',{'reason':'no room reference'}
        reference,mask=self.check_anchor
        current,_=self._anchor(frame,self.screen)
        # Removing screen light changes camera exposure. Fit one global gain and
        # offset to room pixels; camera movement leaves a spatial residual instead.
        before=reference[mask];after=current[mask]
        variance=float(np.var(before)) if len(before) else 0
        gain=float(np.mean((before-before.mean())*(after-after.mean()))/variance) if variance>64 else 0
        offset=float(after.mean()-gain*before.mean()) if len(before) else 0
        residual=float(np.mean(np.abs(after-(gain*before+offset)))) if len(before) else 255
        if mask.sum()<300 or not .4<gain<2.5 or residual>12:
            self.off_allowed=False
            return 'unknown',{'reason':'camera or room changed during power check','room_residual':round(residual,2)}
        stats=self._patch(frame,self.screen)
        # Standby panels reflect ambient light: require both low absolute brightness
        # and a large relative drop, allowing modest reflection gradients.
        if (stats and self.check_lit and stats['p90']<40 and stats['range']<25
                and stats['range']<self.check_lit['range']*.35
                and stats['p90']<self.check_lit['p90']*.4
                and stats['mean']<self.check_lit['mean']*.4):
            return 'off',{'brightness':stats,'room_residual':round(residual,2),'reason':'recent lit screen became persistently dark'}
        return 'unknown',{'brightness':stats,'baseline':self.check_lit,'room_residual':round(residual,2),'reason':'screen state is inconclusive'}
