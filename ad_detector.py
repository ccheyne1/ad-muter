"""Automatically locate the screen and its ad marker. IR output is opt-in."""
import argparse
import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass,field
from collections import deque
from adaptive_vision import Detector,is_ad_token


@dataclass
class Gate:
    """Strict positive confirmation; bounded negative evidence tolerates brief gaps."""
    active: bool = False
    candidate: bool | None = None
    since: float = 0
    count: int = 0
    last: float | None = None
    negatives: deque = field(default_factory=lambda:deque(maxlen=64),repr=False)

    def reset_evidence(self, now):
        """Power events and genuine camera motion invalidate pending observations."""
        self.candidate=None;self.count=0;self.negatives.clear();self.last=now

    def _negative_sample(self, value, now):
        self.negatives.append((now,value))
        while self.negatives and now-self.negatives[0][0]>12:
            self.negatives.popleft()
        clear=[stamp for stamp,reading in self.negatives if reading is False]
        if not clear:self.negatives.clear()
        self.candidate=False if clear else None
        self.count=len(clear)
        if clear:self.since=clear[0]
        return clear

    def observe(self, result, now):
        if result.get('geometry_changed') and result['ad'] is not None:
            # Reset accumulated observations without changing the actual mute state.
            self.reset_evidence(now)
        return self.update(result['ad'],now)

    def update(self, value, now):
        if self.last is not None and (now < self.last or now-self.last > 5):
            self.reset_evidence(now)
        self.last = now
        if self.active:
            if value is True:
                self.reset_evidence(now)
                return None
            clear=self._negative_sample(value,now)
            # Four clear frames across >=3 seconds, all within 12 seconds.
            # Uncertainty may occupy at most half the window; the last two
            # observations must still be clear. Never unmute on unknown alone.
            if (value is False and len(clear)>=4 and now-clear[0]>=3
                    and len(clear)/len(self.negatives)>=.5
                    and len(self.negatives)>=2 and self.negatives[-2][1] is False):
                self.active=False;self.reset_evidence(now)
                return 'unmute'
            return None
        if value is None or value == self.active:
            self.candidate = None
            self.count = 0
            return None
        if value != self.candidate:
            self.candidate, self.since, self.count = value, now, 0
        self.count += 1
        n, seconds = (3, 1.0)
        if self.count >= n and now-self.since >= seconds:
            self.active = value
            self.candidate, self.count = None, 0
            return 'mute' if value else 'unmute'
        return None


def camera_worker(queue, stop):
    """Separate process permits timeout recovery without stale-frame decisions."""
    import queue as q
    from picamera2 import Picamera2
    try:
        with Picamera2() as camera:
            # Picamera2 RGB888 arrays are BGR, matching OpenCV.
            camera.configure(camera.create_still_configuration(main={'size':(4608,2592),'format':'RGB888'},buffer_count=2))
            camera.start();time.sleep(2)
            while not stop.is_set():
                frame=camera.capture_array()
                item=(time.monotonic(),frame,None)
                try:queue.put_nowait(item)
                except q.Full:
                    try:queue.get_nowait()
                    except q.Empty:pass
                    try:queue.put_nowait(item)
                    except q.Full:pass
                stop.wait(.4)
    except Exception as e:
        try:queue.put((time.monotonic(),None,str(e)),timeout=1)
        except q.Full:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',default=str(Path(__file__).with_name('hulu-profile.json')))
    parser.add_argument('--image')
    parser.add_argument('--diagnostics',default='diagnostics/latest')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--idle-timeout',type=float,default=0,help='Stop after this many seconds of a continuously flat screen; 0 disables')
    parser.add_argument('--power-managed',action='store_true',help='Verify screen state after Power; read subsequent power events from stdin')
    parser.add_argument('--save-live-diagnostics',action='store_true',help='Update latest camera and OCR diagnostics every 15 seconds')
    parser.add_argument('--send-ir',action='store_true',help='Actually toggle mute; TV must start unmuted')
    parser.add_argument('--tv-starts-unmuted',action='store_true')
    parser.add_argument('--ir-device',help='Explicit verified transmitter, e.g. /dev/lirc0')
    parser.add_argument('--ir-signal',help='Verified raw mute timing file')
    args=parser.parse_args()
    if args.send_ir and not(args.tv_starts_unmuted and args.ir_device and args.ir_signal):
        parser.error('IR requires --tv-starts-unmuted, --ir-device and --ir-signal')
    if args.seconds<=0:parser.error('--seconds must be positive')
    if args.idle_timeout<0:parser.error('--idle-timeout cannot be negative')
    detector=Detector(json.loads(Path(args.profile).read_text()))
    if args.image:
        result=detector.inspect(detector.cv.imread(args.image),args.diagnostics)
        print(json.dumps(result));return
    if not args.live:parser.error('Choose --image FILE or --live')
    import multiprocessing as mp
    import queue
    context=mp.get_context('spawn');frames=context.Queue(maxsize=1);stop=context.Event()
    worker=context.Process(target=camera_worker,args=(frames,stop),daemon=True)
    gate=Gate();started=time.monotonic();blank_since=None;last_diagnostic=-float('inf')
    if args.power_managed:
        import sys
        import threading
        from tv_power import PowerCheck,ScreenPowerObserver
        power=PowerCheck();observer=ScreenPowerObserver(locator=detector.locator);events=queue.Queue()
        power.request(started);observer.request(False)
        def read_power():
            for line in sys.stdin:
                if line.strip()=='power':events.put(time.monotonic())
        threading.Thread(target=read_power,daemon=True).start()
    worker.start()
    try:
        while time.monotonic()-started<args.seconds:
            try:stamp,frame,error=frames.get(timeout=10)
            except queue.Empty:raise RuntimeError('No fresh camera frame for 10 seconds')
            if error:raise RuntimeError(error)
            if time.monotonic()-stamp>3:
                gate.update(None,time.monotonic());continue
            if args.power_managed:
                requested=False
                while True:
                    try:events.get_nowait();requested=True
                    except queue.Empty:break
                if requested:
                    power.request(time.monotonic());observer.request(power.active)
                    gate.reset_evidence(time.monotonic())
                    print(json.dumps({'event':'power_check_started','previously_running':power.active}),flush=True)
                evidence,detail=observer.inspect(frame,pending=power.pending)
                if power.pending:
                    verdict=power.observe(evidence,time.monotonic())
                    gate.reset_evidence(time.monotonic())
                    print(json.dumps({'event':'power_check','evidence':evidence,'verdict':verdict,
                                      'dry_run':not args.send_ir,**detail}),flush=True)
                    if verdict=='off':
                        print(json.dumps({'event':'tv_off_detector_stopped'}),flush=True);break
                    if verdict=='unknown' and not power.active:
                        print(json.dumps({'event':'power_unknown_returning_to_standby'}),flush=True);break
                    if verdict is None:continue
                    print(json.dumps({'event':'detection_resumed','power_state':verdict}),flush=True)
            diagnostic_path=None
            if args.save_live_diagnostics and time.monotonic()-last_diagnostic>=15:
                diagnostic_path=Path(args.diagnostics)
                diagnostic_path.mkdir(parents=True,exist_ok=True)
                detector.cv.imwrite(str(diagnostic_path/'camera.jpg'),frame)
                last_diagnostic=time.monotonic()
            result=detector.inspect(frame,diagnostic_path) if diagnostic_path else detector.inspect(frame)
            if args.power_managed and not events.empty():
                # A press received during OCR must suppress that frame's mute decision.
                gate.reset_evidence(time.monotonic());continue
            if result.get('blank_screen') and not args.power_managed:
                if blank_since is None:blank_since=time.monotonic()
                if args.idle_timeout and time.monotonic()-blank_since>=args.idle_timeout:
                    print('Screen stayed blank; returning to IR standby.',flush=True);break
            else:blank_since=None
            age=time.monotonic()-stamp
            if age>3:result={'ad':None,'reason':'processing finished on a stale frame'}
            event=gate.observe(result,time.monotonic())
            result.update(event=event,dry_run=not args.send_ir,frame_age_seconds=round(age,3),elapsed_seconds=round(time.monotonic()-started,2),
                          assumed_muted=gate.active,confirmation_count=gate.count,confirmation_candidate=gate.candidate)
            print(json.dumps(result),flush=True)
            if event and args.send_ir:
                # A failed send leaves TV state uncertain: exit; never retry a toggle.
                sent_at=time.monotonic()
                subprocess.run(['ir-ctl','-d',args.ir_device,'--carrier=38000','--duty-cycle=33','--send='+args.ir_signal],check=True,timeout=5)
                print(json.dumps({'event':'ir_transmission_completed','action':event,'return_code':0,
                                  'duration_seconds':round(time.monotonic()-sent_at,3),
                                  'tv_acknowledged':False,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    finally:
        stop.set();worker.join(timeout=3)
        if worker.is_alive():worker.terminate();worker.join(timeout=2)
        frames.close()
        if gate.active:print('Stopped during an ad. TV may remain muted; no exit toggle was sent.',flush=True)


if __name__=='__main__':main()
