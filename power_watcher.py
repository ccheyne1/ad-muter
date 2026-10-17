"""Listen for learned Power; let one camera process verify on/off before OCR."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time


def decode_nec(pairs):
    """Signed microseconds, wire bytes LSB-first. Reject noise and repeat frames."""
    if len(pairs)!=67 or not 7500<=pairs[0]<=10500 or not -5500<=pairs[1]<=-3500:
        return None
    bits=[]
    for i in range(32):
        pulse,space=pairs[2+2*i:4+2*i]
        if not 350<=pulse<=850:return None
        if -850<=space<=-300:bits.append(0)
        elif -2100<=space<=-1250:bits.append(1)
        else:return None
    if not 350<=pairs[-1]<=850:return None
    wire=tuple(sum(bits[i+j]<<j for j in range(8)) for i in range(0,32,8))
    return wire if wire[2]^wire[3]==255 else None


class Frames:
    def __init__(self):self.pairs=[]
    def feed(self,word):
        kind=word&0xff000000;duration=word&0xffffff
        if kind==0x03000000 or (kind==0 and duration>10000):
            result=decode_nec(self.pairs);self.pairs=[];return result
        if kind==0x01000000:
            # Recover when a previous frame ended without a timeout marker.
            if duration>7500:self.pairs=[]
            self.pairs.append(duration)
        elif kind==0:self.pairs.append(-duration)
        else:self.pairs=[]
        if len(self.pairs)>67:self.pairs=[]
        return None


class Debounce:
    def __init__(self,quiet=1.5):self.last=None;self.quiet=quiet
    def accept(self,now):
        accepted=self.last is None or now-self.last>=self.quiet
        self.last=now # Holding a button never becomes another press.
        return accepted


def receiver_path():
    candidates=[]
    for rc in Path('/sys/class/rc').glob('rc*'):
        if 'DRV_NAME=gpio_ir_recv' in (rc/'uevent').read_text():
            candidates.extend(Path('/dev')/p.name for p in rc.glob('lirc*') if p.name[4:].isdigit())
    if len(candidates)!=1:raise RuntimeError(f'Expected one gpio_ir_recv device, found {candidates}')
    return candidates[0]


def log(event,**values):print(json.dumps({'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'event':event,**values}),flush=True)


def notify_power(child):
    """Deliver a check request to the existing camera owner."""
    try:
        child.stdin.write('power\n');child.stdin.flush()
        return True
    except (BrokenPipeError,OSError,ValueError):
        return False


def close_input(child):
    try:child.stdin.close()
    except (BrokenPipeError,OSError,ValueError):pass


def detector_command(send_ir=False,ir_device=None,ir_signal=None):
    command=[sys.executable,str(Path(__file__).with_name('ad_detector.py')),
             '--live','--seconds','86400','--power-managed','--save-live-diagnostics']
    if send_ir:
        if not ir_device or not ir_signal:raise ValueError('Live IR requires device and mute signal')
        command.extend(['--send-ir','--tv-starts-unmuted','--ir-device',ir_device,'--ir-signal',ir_signal])
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('power-watcher.json')))
    parser.add_argument('--listen-only',action='store_true')
    parser.add_argument('--seconds',type=float,default=0,help='0 means indefinitely')
    parser.add_argument('--start-now',action='store_true',help='Check the already-on TV without transmitting Power')
    parser.add_argument('--live-session',action='store_true',help='Enable IR for the first session only; subsequent sessions are dry-run')
    parser.add_argument('--tv-starts-unmuted',action='store_true')
    parser.add_argument('--ir-device')
    parser.add_argument('--ir-signal')
    args=parser.parse_args()
    if args.live_session and (args.listen_only or not args.start_now or not args.tv_starts_unmuted
                             or not args.ir_device or not args.ir_signal):
        parser.error('--live-session requires --start-now, --tv-starts-unmuted, --ir-device and --ir-signal')
    config=json.loads(Path(args.config).read_text())
    target=tuple(int(x,16) for x in config['power_wire_bytes'])
    if len(target)!=4 or any(not 0<=x<=255 for x in target) or target[2]^target[3]!=255:
        raise ValueError('Invalid power wire bytes')
    import fcntl
    lock=open(Path(__file__).with_name('power-watcher.lock'),'w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receiver=receiver_path();fd=os.open(receiver,os.O_RDONLY|os.O_NONBLOCK)
    # LIRC_SET_REC_MODE, LIRC_MODE_MODE2: fixed Linux UAPI unsigned int ioctl.
    fcntl.ioctl(fd,0x40046912,struct.pack('I',4))
    stopping=False;child=None
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    frames=Frames();debounce=Debounce();started=time.monotonic()
    live_available=args.live_session
    def launch():
        nonlocal live_available
        live=live_available;live_available=False
        process=subprocess.Popen(detector_command(live,args.ir_device,args.ir_signal),
                                 cwd=Path(__file__).parent,start_new_session=True,stdin=subprocess.PIPE,text=True)
        log('camera_power_probe_started',pid=process.pid,dry_run=not live)
        return process
    log('listening',device=str(receiver),power_bytes=config['power_wire_bytes'],listen_only=args.listen_only)
    try:
        if args.start_now and not args.listen_only:child=launch()
        while not stopping and (not args.seconds or time.monotonic()-started<args.seconds):
            if child is not None and child.poll() is not None:
                log('detector_exited',returncode=child.returncode)
                close_input(child);child=None
            if not select.select([fd],[],[],.5)[0]:continue
            data=os.read(fd,4096)
            if not data:raise RuntimeError('Receiver disconnected')
            if len(data)%4:raise RuntimeError('Invalid LIRC data length')
            for (word,) in struct.iter_unpack('I',data):
                code=frames.feed(word)
                if code is None:continue
                log('ir_frame',wire_bytes=[f'{x:02x}' for x in code])
                if code!=target or not debounce.accept(time.monotonic()):continue
                if args.listen_only:log('power_observed');continue
                if child is not None and child.poll() is None:
                    if notify_power(child):
                        log('power_check_requested',pid=child.pid)
                        continue
                    log('power_check_delivery_failed',pid=child.pid)
                    # Reap the old process and release its camera before starting another.
                    child.wait(timeout=12)
                if child is not None:close_input(child)
                child=launch()
    finally:
        os.close(fd)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if child is not None:close_input(child)
        lock.close()


if __name__=='__main__':main()
