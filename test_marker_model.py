"""Real training-view transformations and hard negatives; held-out images are excluded."""
from pathlib import Path
import json
import tempfile
import unittest
import cv2
import numpy as np
from adaptive_vision import Detector
from marker_model import MarkerModel,_uint8_percentiles
from unittest.mock import Mock,patch
import time
from screen_locator import Screen
from test_adaptive_vision import screen_scene

ROOT=Path(__file__).parent
PROFILE={'marker_model':'models/hulu-marker-v1.npz','search_corners':['top_right']}


class MarkerTests(unittest.TestCase):
    def test_recent_geometry_hint_expires_and_does_not_skip_current_localization(self):
        detector=Detector(PROFILE)
        screen=Screen(np.array([1450.,150.]),np.array([1.,0.]),np.array([0.,1.]),
                      1300.,700.,.95,'bezel',np.float32([[150,150],[1450,150],[1450,850],[150,850]]))
        detector.previous=screen;detector.previous_at=10
        detector.locator.locate=Mock(return_value=(None,'multiple plausible screens'))
        detector.marker_model.inspect=Mock(return_value={'accepted':False,'uncertain':False,'score':0.,'margin':0.})
        with patch('adaptive_vision.time.monotonic',return_value=14):
            self.assertIsNone(detector.inspect(screen_scene())['ad'])
        self.assertIs(detector.locator.locate.call_args.kwargs['previous'],screen)
        with patch('adaptive_vision.time.monotonic',return_value=16):
            self.assertIsNone(detector.inspect(screen_scene())['ad'])
        self.assertEqual(detector.locator.locate.call_args.kwargs,{})

    @unittest.skipUnless((ROOT/'private-fixtures/show-before-content-panel.jpg').exists() and (ROOT/'private-fixtures/show-content-panel.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_show_frame_with_content_rectangle_keeps_recent_actual_tv_bezel(self):
        first=cv2.imread(str(ROOT/'private-fixtures/show-before-content-panel.jpg'))
        ambiguous=cv2.imread(str(ROOT/'private-fixtures/show-content-panel.jpg'))
        detector=Detector(PROFILE)
        self.assertIs(detector.inspect(first)['ad'],False)
        result=detector.inspect(ambiguous)
        self.assertIs(result['ad'],False,result)
        self.assertTrue(result['screen']['tracking_used'])
        self.assertFalse(result['geometry_changed'])
        self.assertIsNone(Detector(PROFILE).inspect(ambiguous)['ad'])

    def test_histogram_percentiles_preserve_contrast_and_brightness_thresholds(self):
        rng=np.random.default_rng(17)
        samples=[np.zeros((8,8),np.uint8),np.full((9,9),255,np.uint8),
                 np.tile(np.array([49,50,65,74,75,90],np.uint8),(7,1)),
                 np.arange(256,dtype=np.uint8).reshape(16,16)]
        samples.extend(rng.integers(0,256,(h,w),dtype=np.uint8)
                       for h,w in [(1,1),(1,2),(3,7),(70,184),(29,41)])
        samples.append(samples[-1][::2,::2])  # Noncontiguous camera patches.
        for image in samples:
            with self.subTest(shape=image.shape):
                np.testing.assert_array_equal(_uint8_percentiles(image,[5,10,50,90]),
                                              np.percentile(image,[5,10,50,90]))

    def test_expired_processing_budget_never_confirms_absence(self):
        model=MarkerModel(ROOT/'models/hulu-marker-v1.npz')
        result=model.inspect(screen_scene(),deadline=time.monotonic()-1)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['uncertain'])
        self.assertTrue(result['budget_exhausted'])

    def test_camera_bump_after_one_lost_view_clears_old_geometry_evidence(self):
        detector=Detector(PROFILE)
        first=Screen(np.array([1450.,150.]),np.array([1.,0.]),np.array([0.,1.]),
                     1300.,700.,.95,'bezel',np.float32([[150,150],[1450,150],[1450,850],[150,850]]))
        moved=Screen(first.corner+np.array([-200.,80.]),first.across,first.down,
                     first.span,first.vertical_span,first.score,first.method,first.outline+np.array([-200.,80.]))
        detector.locator.locate=Mock(side_effect=[(first,None),(None,'screen not found'),(moved,None)])
        detector.marker_model.inspect=Mock(return_value={'accepted':False,'uncertain':False,'score':0.,'margin':0.})
        self.assertFalse(detector.inspect(screen_scene())['geometry_changed'])
        self.assertIsNone(detector.inspect(screen_scene())['ad'])
        self.assertTrue(detector.inspect(screen_scene())['geometry_changed'])

    @unittest.skipUnless((ROOT/'private-fixtures/development-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_development_marker_follows_camera_translation_rotation_and_scale(self):
        frame=cv2.imread(str(ROOT/'private-fixtures/development-ad.jpg'))
        h,w=frame.shape[:2]
        transforms=[np.float32([[1,0,300],[0,1,160]]),np.float32([[1,0,-250],[0,1,200]]),
                    cv2.getRotationMatrix2D((w/2,h/2),8,1),cv2.getRotationMatrix2D((w/2,h/2),-8,1),
                    cv2.getRotationMatrix2D((w/2,h/2),0,.8)]
        for transform in transforms:
            moved=cv2.warpAffine(frame,transform,(w,h),borderValue=(80,90,100))
            with self.subTest(transform=transform.tolist()):
                result=Detector(PROFILE).inspect(moved)
                self.assertTrue(result['ad'],result)

    def test_lookalike_words_do_not_trigger_learned_badge(self):
        for word in ('Add','bad','Ad123','Advertisement','Ac','AC','AA','00','1:05'):
            frame=screen_scene()
            cv2.rectangle(frame,(1300,175),(1420,215),(220,220,220),-1)
            cv2.putText(frame,word,(1306,204),cv2.FONT_HERSHEY_SIMPLEX,.8 if len(word)<5 else .30,(10,10,10),2)
            with self.subTest(word=word):
                self.assertIsNot(Detector(PROFILE).inspect(frame)['ad'],True)

    @unittest.skipUnless((ROOT/'private-fixtures/development-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_blank_offscreen_or_severely_blurred_views_cannot_confirm_absence(self):
        for value in (0,90,255):
            result=Detector(PROFILE).inspect(np.full((600,1000,3),value,np.uint8))
            self.assertIsNone(result['ad'])
        frame=cv2.imread(str(ROOT/'private-fixtures/development-ad.jpg'))
        self.assertIsNone(Detector(PROFILE).inspect(cv2.GaussianBlur(frame,(101,101),30))['ad'])

    @unittest.skipUnless((ROOT/'private-fixtures/ambiguous-bezel-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_repeated_positive_marker_can_confirm_mute_even_if_bezel_is_ambiguous(self):
        from ad_detector import Gate
        frame=cv2.imread(str(ROOT/'private-fixtures/ambiguous-bezel-ad.jpg'))
        detector=Detector(PROFILE);gate=Gate();events=[]
        for now in (0,.7,1.4,2.1):events.append(gate.observe(detector.inspect(frame),now))
        self.assertEqual(events.count('mute'),1)

    @unittest.skipUnless((ROOT/'private-fixtures/faint-ad.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_partial_native_glyphs_do_not_depend_on_a_low_confidence_screen_crop(self):
        frame=cv2.imread(str(ROOT/'private-fixtures/faint-ad.jpg'))
        result=Detector(PROFILE).inspect(frame)
        self.assertTrue(result['ad'],result)
        self.assertLess(result['screen']['score'],.8)

    def test_corrupt_model_is_rejected(self):
        root=ROOT/'diagnostics/test-runs';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path=Path(directory)/'bad.npz'
            np.savez(path,templates=np.full((1,2772),np.nan),negatives=np.zeros((1,2772)),threshold=.68,margin=.08,feature_version=3)
            with self.assertRaises(ValueError):MarkerModel(path)


if __name__=='__main__':unittest.main()
