"""Geometry recovery and conservative decision tests; no camera or IR required."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from adaptive_vision import Detector
from screen_locator import ScreenLocator,rectify_header,Screen,_tracked_screen

ROOT=Path(__file__).parent


def screen_scene(two=False):
    image=np.full((900,1600,3),(80,90,100),np.uint8)
    rectangles=[(90,130,730,550),(870,130,1510,550)] if two else [(150,150,1450,850)]
    for x0,y0,x1,y1 in rectangles:
        cv2.rectangle(image,(x0,y0),(x1,y1),(5,5,5),-1)
        cv2.rectangle(image,(x0+10,y0+10),(x1-10,y1-10),(125,90,30),-1)
        for y in range(y0+35,y1-10,70):
            cv2.line(image,(x0+35,y),(x1-35,y),(190,150,70),12)
        cv2.rectangle(image,(x1-150,y0+25),(x1-30,y0+65),(220,220,220),-1)
        cv2.putText(image,'Ad',(x1-140,y0+55),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,10),2)
    return image


def fake_ocr(text='No',confidence=90):
    return {'text':[text],'conf':[confidence],'left':[12],'top':[12],'width':[40],'height':[20]}


class GeometryTests(unittest.TestCase):
    def test_recent_display_can_be_followed_when_another_display_appears(self):
        two=screen_scene(two=True)
        one=two.copy();one[:,:800]=(80,90,100)
        locator=ScreenLocator();prior,reason=locator.locate(one)
        self.assertIsNotNone(prior,reason)
        self.assertIsNone(locator.locate(two)[0])
        current,reason=locator.locate(two,previous=prior)
        self.assertIsNotNone(current,reason)
        self.assertTrue(current.tracking_used)
        self.assertLess(float(np.linalg.norm(current.corner-prior.corner)),10)
        self.assertIsNone(locator.locate(np.zeros_like(two),previous=prior)[0])

    def test_tracking_requires_one_strong_current_bezel_and_rejects_camera_motion(self):
        prior=Screen(np.array([1400.,150.]),np.array([1.,0.]),np.array([0.,1.]),
                     1000.,600.,.95,'bezel',np.float32([[400,150],[1400,150],[1400,750],[400,750]]))
        def candidate(offset,score=.95):
            delta=np.array(offset,dtype=float)
            return Screen(prior.corner+delta,prior.across,prior.down,prior.span,prior.vertical_span,
                          score,prior.method,prior.outline+delta)
        current=candidate([8,5]);other=candidate([-500,40])
        self.assertIs(_tracked_screen([current,other],prior,(900,1600,3)),current)
        self.assertIsNone(_tracked_screen([current,candidate([15,10])],prior,(900,1600,3)))
        self.assertIsNone(_tracked_screen([candidate([200,80]),other],prior,(900,1600,3)))
        self.assertIsNone(_tracked_screen([candidate([8,5],.80),other],prior,(900,1600,3)))
        self.assertIsNone(_tracked_screen([],prior,(900,1600,3)))
        self.assertIsNone(_tracked_screen([current],None,(900,1600,3)))

    def test_partial_bezel_length_jitter_is_not_camera_motion(self):
        first=Screen(np.array([2800.,30.]),np.array([1.,0.]),np.array([.4,.9165]),
                     1400.,2000.,.95,'bezel',np.float32([[1400,30],[2800,30],[3700,2100]]))
        changed=Screen(first.corner.copy(),first.across.copy(),np.array([.76,.65]),
                       2800.,2200.,.96,'bezel',first.outline.copy())
        self.assertFalse(changed.moved_from(first,(2592,4608,3)))
        changed.corner+=np.array([200.,100.])
        self.assertTrue(changed.moved_from(first,(2592,4608,3)))

    def test_complete_screen_scale_change_still_reports_motion(self):
        first=Screen(np.array([1450.,150.]),np.array([1.,0.]),np.array([0.,1.]),
                     1300.,700.,.95,'bezel',np.float32([[150,150],[1450,150],[1450,850],[150,850]]))
        changed=Screen(first.corner.copy(),first.across.copy(),first.down.copy(),
                       1050.,600.,.95,'bezel',np.float32([[400,150],[1450,150],[1450,750],[400,750]]))
        self.assertTrue(changed.moved_from(first,(900,1600,3)))

    @unittest.skipUnless((ROOT/'private-fixtures/reference-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_reference_label_is_inside_dynamic_region_after_camera_motion(self):
        original=cv2.imread(str(ROOT/'private-fixtures/reference-ad.jpg'))
        self.assertIsNotNone(original)
        h,w=original.shape[:2]
        label=np.float32([2530,990,1])
        transformations=[np.float32([[1,0,0],[0,1,0]]),np.float32([[1,0,200],[0,1,-120]]),
                         np.float32([[1,0,-300],[0,1,140]]),cv2.getRotationMatrix2D((w/2,h/2),8,1),
                         cv2.getRotationMatrix2D((w/2,h/2),-8,1)]
        for transform in transformations:
            with self.subTest(transform=transform.tolist()):
                moved=cv2.warpAffine(original,transform,(w,h),borderValue=(95,100,105))
                screen,reason=ScreenLocator().locate(moved)
                self.assertIsNotNone(screen,reason)
                polygon=screen.header_quad()
                current=transform@label
                self.assertGreaterEqual(cv2.pointPolygonTest(polygon,tuple(map(float,current)),False),0)

    def test_full_screen_has_projective_quad(self):
        image=screen_scene()
        screen,reason=ScreenLocator().locate(image)
        self.assertIsNotNone(screen,reason)
        quad=screen.full_quad(image.shape)
        self.assertIsNotNone(quad)
        np.testing.assert_allclose(quad,[[150,150],[1450,150],[1450,850],[150,850]],atol=3)

    def test_two_similar_displays_are_ambiguous(self):
        screen,reason=ScreenLocator().locate(screen_scene(two=True))
        self.assertIsNone(screen)
        self.assertEqual(reason,'multiple plausible screens')

    @unittest.skipUnless((ROOT/'private-fixtures/framed-content.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_framed_content_inside_real_roku_screen_is_not_another_tv(self):
        frame=cv2.imread(str(ROOT/'private-fixtures/framed-content.jpg'))
        self.assertIsNotNone(frame)
        screen,reason=ScreenLocator().locate(frame)
        self.assertIsNotNone(screen,reason)
        self.assertGreater(screen.score,.9)
        self.assertGreater(screen.corner[0],2400)

    def test_blank_frame_has_no_screen(self):
        for value in (0,80,255):
            screen,reason=ScreenLocator().locate(np.full((600,1000,3),value,np.uint8))
            self.assertIsNone(screen)
            self.assertEqual(reason,'frame too flat or dark')

    @unittest.skipUnless((ROOT/'private-fixtures/reference-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_visible_partial_tv_does_not_invent_fourth_corner(self):
        frame=cv2.imread(str(ROOT/'private-fixtures/reference-ad.jpg'))
        screen,reason=ScreenLocator().locate(frame)
        self.assertIsNotNone(screen,reason)
        self.assertIsNone(screen.full_quad(frame.shape))
        _,valid,_=rectify_header(frame,screen)
        self.assertGreater(np.mean(valid>0),.98)


@unittest.skipUnless((ROOT/'private-fixtures/reference-ad.jpg').exists(), 'Private reference photo is not distributed')
class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.frame=cv2.imread(str(ROOT/'private-fixtures/reference-ad.jpg'))
        self.detector=Detector({'ocr_budget_seconds':5})

    def test_movement_is_reported_without_using_reference_coordinates(self):
        self.detector._ocr_callable=lambda image,psm:fake_ocr()
        first=self.detector.inspect(self.frame)
        h,w=self.frame.shape[:2]
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,180],[0,1,-100]]),(w,h),borderValue=(95,100,105))
        second=self.detector.inspect(moved)
        self.assertFalse(first['geometry_changed'])
        self.assertTrue(second['geometry_changed'])
        json.dumps(second) # Real movement must remain serializable in the live JSON log.
        self.assertGreater(np.linalg.norm(np.array(first['screen']['corner'])-second['screen']['corner']),150)

    def test_lost_screen_is_unknown_and_clears_stale_diagnostics(self):
        self.detector._ocr_callable=lambda image,psm:fake_ocr('Ad')
        temporary_root=ROOT/'diagnostics'/'test-runs'
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(ROOT.resolve()))
            positive=self.detector.inspect(self.frame,directory)
            self.assertTrue(positive['ad'])
            self.assertTrue((Path(directory)/'badge.png').exists())
            lost=self.detector.inspect(np.zeros_like(self.frame),directory)
            self.assertIsNone(lost['ad'])
            self.assertFalse((Path(directory)/'badge.png').exists())
            self.assertFalse((Path(directory)/'screen.jpg').exists())
            self.assertIsNone(json.loads((Path(directory)/'result.json').read_text())['ad'])

    def test_ocr_failure_cannot_become_negative_evidence(self):
        def failed(image,psm):raise RuntimeError('Tesseract timed out')
        self.detector._ocr_callable=failed
        result=self.detector.inspect(self.frame)
        self.assertIsNone(result['ad'])
        self.assertIn('timeout',result['reason'])

    def test_unavailable_requested_corner_cannot_become_negative(self):
        detector=Detector({'search_corners':['top_right','top_left']})
        detector._ocr_callable=lambda image,psm:fake_ocr()
        result=detector.inspect(self.frame)
        self.assertIsNone(result['ad'])
        self.assertEqual(result['unavailable_corners'],['top_left'])

    def test_legacy_pixel_coordinates_have_no_effect(self):
        a=Detector({'badge_quad':[[0,0]]*4,'reference_size':[1,2]})
        b=Detector()
        a._ocr_callable=b._ocr_callable=lambda image,psm:fake_ocr('Ad')
        ra,rb=a.inspect(self.frame),b.inspect(self.frame)
        self.assertEqual(ra['screen'],rb['screen'])
        self.assertEqual(ra['ad'],rb['ad'])

    def test_invalid_frames_are_unknown(self):
        for frame in (None,np.zeros((300,300),np.uint8),np.zeros((30,30,3),np.uint8)):
            self.assertIsNone(self.detector.inspect(frame)['ad'])
    def test_lookalike_ocr_can_never_mute(self):
        self.detector._ocr_callable=lambda image,psm:fake_ocr('Ac')
        self.assertIsNone(self.detector.inspect(self.frame)['ad'])
    def test_only_requested_corner_is_searched(self):
        detector=Detector({'search_corners':['top_left']})
        detector._ocr_callable=lambda image,psm:fake_ocr('Ad')
        self.assertIsNone(detector.inspect(self.frame)['ad'])


if __name__=='__main__':unittest.main()
