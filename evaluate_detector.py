"""End-to-end adaptive OCR regressions; synthetic changes are not independent footage."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from ad_detector import Detector


def cases(image, negative=None):
    h,w=image.shape[:2]
    result=[('original',image,True)]
    for name,dx,dy in [('shift_right',180,-100),('shift_left',-300,120),('shift_down',80,210)]:
        transform=np.float32([[1,0,dx],[0,1,dy]])
        result.append((name,cv2.warpAffine(image,transform,(w,h),borderValue=(95,100,105)),True))
    for angle in (-8,8):
        transform=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        result.append((f'roll_{angle}',cv2.warpAffine(image,transform,(w,h),borderValue=(95,100,105)),True))
    transform=cv2.getRotationMatrix2D((w/2,h/2),0,.80)
    result.append(('zoom_out',cv2.warpAffine(image,transform,(w,h),borderValue=(95,100,105)),True))
    source=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    destination=np.float32([[w*.06,h*.08],[w*.94,0],[w-1,h*.96],[0,h*.94]])
    result.append(('perspective_change',cv2.warpPerspective(image,cv2.getPerspectiveTransform(source,destination),(w,h),borderValue=(95,100,105)),True))
    result.append(('different_frame_aspect',cv2.copyMakeBorder(image,140,50,180,300,cv2.BORDER_CONSTANT,value=(95,100,105)),True))
    result.append(('dimmer',cv2.convertScaleAbs(image,alpha=.70),True))
    result.append(('mild_blur',cv2.GaussianBlur(image,(3,3),.5),True))
    # Source-specific test ground truth; this polygon is never read by the detector.
    quad=np.float32([[2476,966],[2597,965],[2598,1008],[2475,1012]])*[w/4608,h/2592]
    mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,quad.astype(np.int32),255)
    erased=cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
    result.append(('label_removed',erased,False))
    x0,y0,x1,y1=[round(v) for v in (2470*w/4608,958*h/2592,2602*w/4608,1016*h/2592)]
    moved_label=erased.copy();patch=image[y0:y1,x0:x1]
    nx,ny=x0-round(210*w/4608),y0+round(12*h/2592)
    moved_label[ny:ny+patch.shape[0],nx:nx+patch.shape[1]]=patch
    result.append(('marker_relocated_inside_screen',moved_label,True))
    if negative is not None:
        result.append(('real_roku_menu',negative,False));nh,nw=negative.shape[:2]
        result.append(('shifted_roku_menu',cv2.warpAffine(negative,np.float32([[1,0,-200],[0,1,100]]),(nw,nh),borderValue=(95,100,105)),False))
    # Independently drawn marker fonts and fully visible screens exercise the other path.
    from test_adaptive_vision import screen_scene
    for word,expected in [('Ad',True),('Add',False),('bad',False),('Advertisement',False)]:
        scene=screen_scene()
        cv2.rectangle(scene,(1300,175),(1420,215),(220,220,220),-1)
        scale=.8 if len(word)<4 else .30
        cv2.putText(scene,word,(1306,204),cv2.FONT_HERSHEY_SIMPLEX,scale,(10,10,10),2)
        result.append(('full_screen_word_'+word,scene,expected))
    transparent=screen_scene();cv2.rectangle(transparent,(1300,175),(1420,215),(125,90,30),-1)
    cv2.putText(transparent,'Ad',(1310,205),cv2.FONT_HERSHEY_SIMPLEX,.85,(245,245,245),2)
    result.append(('plain_text_overlay',transparent,True))
    left=screen_scene();cv2.rectangle(left,(1300,175),(1420,215),(125,90,30),-1)
    cv2.rectangle(left,(180,175),(300,215),(220,220,220),-1)
    cv2.putText(left,'Ad',(190,205),cv2.FONT_HERSHEY_SIMPLEX,.85,(10,10,10),2)
    result.append(('all_corners_top_left_marker',left,True))
    result.extend([('blocked_black',np.zeros_like(image),None),('blank_white',np.full_like(image,255),None),
                   ('unreadably_blurred',cv2.GaussianBlur(image,(51,51),15),None)])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',help='User-supplied reference image with its Ad label')
    parser.add_argument('--negative',help='Real camera photo with no ad marker')
    parser.add_argument('--extra-negative',action='append',default=[],help='Additional real negative photos')
    parser.add_argument('--out',default='diagnostics/adaptive/regression.json')
    parser.add_argument('--profile',default=str(Path(__file__).with_name('hulu-profile.json')))
    args=parser.parse_args();image=cv2.imread(args.image)
    negative=cv2.imread(args.negative) if args.negative else None
    if image is None or (args.negative and negative is None):raise ValueError('Image could not be loaded')
    profile=json.loads(Path(args.profile).read_text());report=[]
    evaluations=cases(image,negative)
    for number,path in enumerate(args.extra_negative):
        frame=cv2.imread(path)
        if frame is None:raise ValueError('Image could not be loaded: '+path)
        evaluations.append((f'extra_real_negative_{number+1}',frame,False))
    for name,frame,expected in evaluations:
        settings=profile if name!='all_corners_top_left_marker' else {**profile,'search_corners':['top_right','top_left','bottom_right','bottom_left']}
        result=Detector(settings).inspect(frame);row={'case':name,'expected':expected,'passed':result['ad'] is expected,**result}
        report.append(row)
        print(json.dumps({k:row.get(k) for k in ('case','expected','ad','passed','reason','processing_seconds')}),flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'source':str(args.image),'scope':'one positive source and synthetic derivatives; not a field accuracy estimate',
                               'passed':sum(row['passed'] for row in report),'total':len(report),'cases':report},indent=2),encoding='utf-8')
    if not all(row['passed'] for row in report):raise SystemExit(1)


if __name__=='__main__':main()
