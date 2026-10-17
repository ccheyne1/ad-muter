import unittest
from ad_detector import Gate,is_ad_token

class Tests(unittest.TestCase):
    def test_words(self):
        for word in ['Ad','AD','(Ad)','‘Ad']:self.assertTrue(is_ad_token(word))
        for word in ['bad','Advertisement','A D','A','Add','read','','ad123','123ad']:
            self.assertFalse(is_ad_token(word))
    def test_confirmation_and_single_toggle(self):
        g=Gate()
        self.assertIsNone(g.update(True,0));self.assertIsNone(g.update(True,.5))
        self.assertEqual(g.update(True,1),'mute')
        for t in [2,3,4]:self.assertIsNone(g.update(True,t))
        for t in [5,5.6,6.2,6.8,7.4]:self.assertIsNone(g.update(False,t))
        self.assertEqual(g.update(False,8.1),'unmute')
    def test_unknown_breaks_evidence(self):
        g=Gate(active=True)
        for t in [0,1,2]:g.update(False,t)
        g.update(None,3)
        self.assertIsNone(g.update(False,4));self.assertTrue(g.active)
    def test_stale_gap_does_not_confirm(self):
        g=Gate();g.update(True,0);g.update(True,.5)
        self.assertIsNone(g.update(True,10));self.assertFalse(g.active)
    def test_false_positive_burst_never_mutes(self):
        g=Gate()
        for t,value in enumerate([True,True,False,True,None,True,False]):
            self.assertIsNone(g.update(value,t))
        self.assertFalse(g.active)
    def test_short_negative_gap_never_unmutes(self):
        g=Gate(active=True)
        for t,value in enumerate([False,False,True,False,None,False,True]):
            self.assertIsNone(g.update(value,t))
        self.assertTrue(g.active)
    def test_camera_bump_cannot_finish_old_unmute_evidence(self):
        g=Gate(active=True)
        for t in [0,.6,1.2,1.8,2.4]:
            self.assertIsNone(g.observe({'ad':False},t))
        self.assertIsNone(g.observe({'ad':False,'geometry_changed':True},3.1))
        self.assertTrue(g.active)
        for t in [3.7,4.3,4.9,5.5]:self.assertIsNone(g.observe({'ad':False},t))
        self.assertEqual(g.observe({'ad':False},6.2),'unmute')
    def test_camera_bump_requires_fresh_positive_confirmation(self):
        g=Gate();g.observe({'ad':True},0);g.observe({'ad':True},.5)
        self.assertIsNone(g.observe({'ad':True,'geometry_changed':True},1))
        self.assertIsNone(g.observe({'ad':True},1.5))
        self.assertEqual(g.observe({'ad':True},2),'mute')
    def test_unmute_tolerates_one_uncertain_frame_but_requires_fresh_clear_end(self):
        g=Gate(active=True)
        for t,value in enumerate([False,False,False,None,False]):
            self.assertIsNone(g.update(value,t))
        self.assertEqual(g.update(False,5),'unmute')
        self.assertIsNone(g.update(False,7))
    def test_unknown_cannot_complete_old_negative_evidence(self):
        g=Gate(active=True)
        for t in [0,.1,.2,.3,.4,.5]:self.assertIsNone(g.update(False,t))
        self.assertIsNone(g.update(None,3))
        self.assertIsNone(g.update(False,3.1))
        self.assertEqual(g.update(False,3.2),'unmute')
    def test_mostly_uncertain_sequence_never_unmutes(self):
        g=Gate(active=True)
        for t in range(40):self.assertIsNone(g.update(False if t%2 else None,t))
        self.assertTrue(g.active)
    def test_long_unknown_interval_expires_all_old_negatives(self):
        g=Gate(active=True)
        for t in range(3):g.update(False,t)
        for t in range(3,25):self.assertIsNone(g.update(None,t))
        self.assertEqual(g.count,0)
        for t in range(25,28):self.assertIsNone(g.update(False,t))
        self.assertEqual(g.update(False,28),'unmute')
    def test_positive_marker_or_power_request_clears_pending_unmute(self):
        for interruption in ('ad','power'):
            with self.subTest(interruption=interruption):
                g=Gate(active=True)
                for t in range(3):g.update(False,t)
                if interruption=='ad':g.update(True,5)
                else:g.reset_evidence(5)
                for t in range(6,9):self.assertIsNone(g.update(False,t))
                self.assertEqual(g.update(False,9),'unmute')

    def test_clear_show_restores_sound_in_three_seconds(self):
        g=Gate(active=True)
        for t in range(3):self.assertIsNone(g.update(False,t))
        self.assertEqual(g.update(False,3),'unmute')

    def test_noisy_stream_needs_enough_clear_recent_evidence(self):
        g=Gate(active=True)
        for t,value in enumerate([False,None,None,False,None,None,False,None,None,False,False]):
            self.assertIsNone(g.update(value,t))
        self.assertTrue(g.active)

    def test_live_latency_regression_without_ir(self):
        import json
        from pathlib import Path
        rows=[]
        for line in (Path(__file__).parent/'tests/fixtures/live-decisions.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if 'ad' in row:rows.append(row)
        g=Gate();events=[]
        for row in rows:
            event=g.observe(row,row['elapsed_seconds'])
            if event:events.append((event,row['elapsed_seconds']))
        self.assertEqual(events,[('mute',205.16),('unmute',325.04)])

if __name__=='__main__':unittest.main()
