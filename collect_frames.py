"""Save human-labeled camera frames for later review. This tool sends no IR."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid


LABELS=('ad','no_ad','unknown')


def save_frame(frame,output,label,app='hulu',batch=None):
    import cv2
    if label not in LABELS:raise ValueError('Use ad, no_ad, or unknown')
    timestamp=datetime.now(timezone.utc)
    directory=Path(output)/label;directory.mkdir(parents=True,exist_ok=True)
    path=directory/(timestamp.strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:8]+'.jpg')
    if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):
        raise RuntimeError('Could not save '+str(path))
    row={'file':path.relative_to(output).as_posix(),'label':label,'label_source':'human',
         'app':app,'batch':batch,'captured_at_utc':timestamp.isoformat(),
         'width':int(frame.shape[1]),'height':int(frame.shape[0]),
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # Append only; never replace a previous label or image.
    with (Path(output)/'manifest.jsonl').open('a',encoding='utf-8') as manifest:
        manifest.write(json.dumps(row)+'\n')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True,choices=LABELS)
    parser.add_argument('--count',type=int,default=3,help='1-30 frames; keep batches short near ad boundaries')
    parser.add_argument('--interval',type=float,default=1,help='Seconds between frames, .5-10')
    parser.add_argument('--app',default='hulu')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'dataset')
    args=parser.parse_args()
    if not 1<=args.count<=30 or not .5<=args.interval<=10:
        parser.error('Use count 1-30 and interval .5-10')
    for unit in ('admuter-live','admuter-power'):
        if subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0:
            parser.error('Stop admuter-live and admuter-power before using the camera')
    from picamera2 import Picamera2
    batch=uuid.uuid4().hex
    print(f'Label: {args.label}. Camera warming up for two seconds. Keep the scene in this category.',flush=True)
    with Picamera2() as camera:
        camera.configure(camera.create_still_configuration(main={'size':(4608,2592),'format':'RGB888'},buffer_count=2))
        camera.start();time.sleep(2)
        for number in range(args.count):
            row=save_frame(camera.capture_array(),args.out,args.label,args.app,batch)
            print(f"Saved {number+1}/{args.count}: {row['file']}",flush=True)
            if number+1<args.count:time.sleep(args.interval)


if __name__=='__main__':main()
