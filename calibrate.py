"""Inspect automatic screen localization; no pixel-coordinate calibration needed."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image')
    p.add_argument('--out',default='diagnostics/screen-check',help='Diagnostic directory')
    args=p.parse_args()
    import cv2
    from ad_detector import Detector
    im=cv2.imread(args.image)
    if im is None:p.error('Cannot read image')
    profile=json.loads(Path(__file__).with_name('hulu-profile.json').read_text())
    detector=Detector(profile)
    result=detector.inspect(im,args.out)
    print(json.dumps(result));print(f'Screen overlay: {Path(args.out)/"screen.jpg"}')

if __name__=='__main__':main()
