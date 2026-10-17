"""Power toggle lifecycle and camera evidence without GPIO, OCR, or a live TV."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from tv_power import PowerCheck,ScreenPowerObserver
from test_adaptive_vision import screen_scene
ROOT=Path(__file__).parent


class LifecycleTests(unittest.TestCase):
    def test_cold_start_waits_for_boot_and_stable_on(self):
        check=PowerCheck();check.request(0)
        for now in (0,1,2,3,10):self.assertIsNone(check.observe('unknown',now))
        for now in (11,11.5):self.assertIsNone(check.observe('on',now))
        self.assertEqual(check.observe('on',12),'on');self.assertTrue(check.active)

    def test_off_check_requires_settle_and_sustained_darkness(self):
        check=PowerCheck();check.active=True;check.request(0)
        for now in (0,2,5,6,7,8,9):self.assertIsNone(check.observe('off',now))
        self.assertEqual(check.observe('off',10),'off');self.assertFalse(check.active)

    def test_second_press_restarts_check_instead_of_flipping_a_flag(self):
        check=PowerCheck();check.active=True;check.request(0)
        check.observe('off',6);check.observe('off',7);check.request(8)
        self.assertTrue(check.active)
        for now in (14,15,16):self.assertIsNone(check.observe('on',now))
        self.assertEqual(check.observe('on',17),'on')

    def test_unknown_timeout_preserves_previous_state(self):
        for active in (True,False):
            check=PowerCheck();check.active=active;check.request(0)
            self.assertEqual(check.observe('unknown',45),'unknown')
            self.assertEqual(check.active,active)

    def test_one_dark_frame_and_missing_frames_cannot_stop(self):
        check=PowerCheck();check.active=True;check.request(0)
        for evidence,now in [('off',6),('unknown',7),('off',8),('off',12),('off',13)]:
            self.assertIsNone(check.observe(evidence,now))
        self.assertTrue(check.active)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.frame=screen_scene()
        # Distinct room detail gives the motion check something to compare.
        for x in range(0,1600,70):
            cv2.rectangle(self.frame,(x,10),(x+25,120),(145,145,145),-1)
        self.dark=self.frame.copy()
        cv2.rectangle(self.dark,(150,150),(1450,850),(4,4,4),-1)

    def test_off_requires_lit_baseline_and_power_request(self):
        observer=ScreenPowerObserver();observer.request(False)
        self.assertEqual(observer.inspect(self.dark,True)[0],'unknown')
        self.assertEqual(observer.inspect(self.frame)[0],'on')
        self.assertEqual(observer.inspect(self.dark)[0],'unknown')
        observer.request(True)
        self.assertEqual(observer.inspect(self.dark,True)[0],'off')

    def test_camera_motion_invalidates_off_reference(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        moved=cv2.warpAffine(self.dark,np.float32([[1,0,160],[0,1,80]]),(1600,900),borderValue=(140,150,160))
        self.assertEqual(observer.inspect(moved,True)[0],'unknown')
        self.assertFalse(observer.off_allowed)

    def test_off_panel_can_have_a_small_room_reflection(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        reflected=self.dark.copy()
        for y in range(150,851):
            reflected[y,150:1451]=18+int(16*(y-150)/700)
        self.assertEqual(observer.inspect(reflected,True)[0],'off')

    def test_global_auto_exposure_change_does_not_look_like_camera_motion(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        brighter_room=cv2.convertScaleAbs(self.dark,alpha=1.3,beta=10)
        self.assertEqual(observer.inspect(brighter_room,True)[0],'off')

    def test_featureless_room_cannot_supply_motion_reference(self):
        observer=ScreenPowerObserver();plain=screen_scene();observer.inspect(plain);observer.request(True)
        cv2.rectangle(plain,(150,150),(1450,850),(4,4,4),-1)
        self.assertEqual(observer.inspect(plain,True)[0],'unknown')

    def test_dim_content_with_contrast_is_not_off(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        dim=self.dark.copy()
        for x in range(150,1451):
            dim[150:851,x]=8 if (x//45)%2 else 65
        self.assertNotEqual(observer.inspect(dim,True)[0],'off')

    def test_shutdown_fade_does_not_replace_pre_press_brightness(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        baseline=dict(observer.check_lit)
        dim=self.frame.copy()
        dim[150:851,150:1451]=cv2.convertScaleAbs(dim[150:851,150:1451],alpha=.6)
        observer.inspect(dim,True)
        self.assertEqual(observer.check_lit,baseline)

    def test_reacquires_moved_lit_screen(self):
        observer=ScreenPowerObserver();observer.inspect(self.frame);observer.request(True)
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,-100],[0,1,-80]]),(1600,900),borderValue=(80,90,100))
        self.assertEqual(observer.inspect(moved,True)[0],'on')
        self.assertFalse(observer.off_allowed)

    @unittest.skipUnless((ROOT/'private-fixtures/reference-ad.jpg').exists() and (ROOT/'private-fixtures/current-tv-view.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_partial_real_tv_supports_on_check(self):
        for name in ('reference-ad.jpg','current-tv-view.jpg'):
            frame=cv2.imread(str(Path(__file__).parent/'private-fixtures'/name))
            with self.subTest(name=name):
                self.assertEqual(ScreenPowerObserver().inspect(frame,True)[0],'on')

    @unittest.skipUnless((ROOT/'private-fixtures/framed-content.jpg').exists() and (ROOT/'private-fixtures/tv-off-reflected.jpg').exists(), 'Private camera fixtures are not distributed')
    def test_real_partial_screen_off_excludes_tv_content_from_room_reference(self):
        root=Path(__file__).parent/'private-fixtures'
        observer=ScreenPowerObserver()
        self.assertEqual(observer.inspect(cv2.imread(str(root/'framed-content.jpg')))[0],'on')
        observer.request(True)
        evidence,detail=observer.inspect(cv2.imread(str(root/'tv-off-reflected.jpg')),True)
        self.assertEqual(evidence,'off',detail)
        self.assertLess(detail['room_residual'],5)

    def test_unrelated_flat_or_bright_room_is_unknown(self):
        for level in (0,90,255):
            observer=ScreenPowerObserver();observer.request(False)
            self.assertEqual(observer.inspect(np.full((900,1600,3),level,np.uint8),True)[0],'unknown')


if __name__=='__main__':unittest.main()
