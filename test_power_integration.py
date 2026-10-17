"""Exercise the live loop: a Power press arriving during OCR suppresses mute."""
import io
import queue
import unittest
import json
import subprocess
from unittest.mock import Mock,patch
import ad_detector


class LiveLoopTests(unittest.TestCase):
    def test_real_output_sends_once_for_each_confirmed_transition(self):
        clock=[0.0];sequence=iter([True]*3+[False]*6)
        frames=Mock();values=[]
        def next_frame(timeout):
            value=next(sequence);values.append(value);clock[0]+=.7
            return clock[0],object(),None
        frames.get.side_effect=next_frame
        context=Mock();context.Queue.return_value=frames
        context.Process.return_value.is_alive.return_value=False
        detector=Mock();detector.inspect.side_effect=lambda frame:{'ad':values[-1]}
        arguments=['ad_detector.py','--live','--send-ir','--tv-starts-unmuted',
                   '--ir-device','/dev/lirc0','--ir-signal','mute.txt']
        output=io.StringIO()
        with (patch('sys.argv',arguments),patch('sys.stdout',output),
              patch('ad_detector.Detector',return_value=detector),
              patch('multiprocessing.get_context',return_value=context),
              patch('ad_detector.time.monotonic',side_effect=lambda:clock[0]),
              patch('ad_detector.subprocess.run') as send):
            with self.assertRaises(StopIteration):ad_detector.main()
        self.assertEqual(send.call_count,2)
        for call in send.call_args_list:
            self.assertEqual(call.args[0],['ir-ctl','-d','/dev/lirc0','--carrier=38000',
                                         '--duty-cycle=33','--send=mute.txt'])
            self.assertTrue(call.kwargs['check'])
        receipts=[json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        receipts=[r for r in receipts if r.get('event')=='ir_transmission_completed']
        self.assertEqual([r['action'] for r in receipts],['mute','unmute'])
        self.assertTrue(all(r['return_code']==0 and not r['tv_acknowledged'] for r in receipts))

    def test_failed_transmission_exits_without_retry_or_success_receipt(self):
        clock=[0.];frames=Mock()
        def next_frame(timeout):
            clock[0]+=.7
            return clock[0],object(),None
        frames.get.side_effect=next_frame
        context=Mock();context.Queue.return_value=frames
        context.Process.return_value.is_alive.return_value=False
        detector=Mock();detector.inspect.return_value={'ad':True}
        output=io.StringIO()
        arguments=['ad_detector.py','--live','--send-ir','--tv-starts-unmuted',
                   '--ir-device','/dev/lirc0','--ir-signal','mute.txt']
        with (patch('sys.argv',arguments),patch('sys.stdout',output),
              patch('ad_detector.Detector',return_value=detector),
              patch('multiprocessing.get_context',return_value=context),
              patch('ad_detector.time.monotonic',side_effect=lambda:clock[0]),
              patch('ad_detector.subprocess.run',side_effect=subprocess.CalledProcessError(1,['ir-ctl'])) as send):
            with self.assertRaises(subprocess.CalledProcessError):ad_detector.main()
        self.assertEqual(send.call_count,1)
        self.assertNotIn('ir_transmission_completed',output.getvalue())
        frames.close.assert_called_once()

    def test_power_received_during_ocr_cancels_ready_mute(self):
        clock=[0.0]
        events=queue.Queue()
        frames=Mock()
        def next_frame(timeout):
            if clock[0]>1.8:raise StopIteration('test completed')
            clock[0]+=.6
            return clock[0],object(),None
        frames.get.side_effect=next_frame
        context=Mock();context.Queue.return_value=frames
        context.Process.return_value.is_alive.return_value=False
        detector=Mock();calls=[0]
        def inspect(frame):
            calls[0]+=1
            if calls[0]==3:events.put(clock[0])
            return {'ad':True}
        detector.inspect.side_effect=inspect
        power=Mock(active=True,pending=False)
        observer=Mock();observer.inspect.return_value=('on',{})
        arguments=['ad_detector.py','--live','--power-managed','--send-ir',
                   '--tv-starts-unmuted','--ir-device','/dev/lirc0','--ir-signal','mute.txt']
        with (patch('sys.argv',arguments),patch('sys.stdin',io.StringIO()),
              patch('sys.stdout',io.StringIO()),patch('ad_detector.Detector',return_value=detector),
              patch('multiprocessing.get_context',return_value=context),
              patch('queue.Queue',return_value=events),patch('threading.Thread'),
              patch('tv_power.PowerCheck',return_value=power),
              patch('tv_power.ScreenPowerObserver',return_value=observer),
              patch('ad_detector.time.monotonic',side_effect=lambda:clock[0]),
              patch('ad_detector.subprocess.run') as send):
            with self.assertRaises(StopIteration):ad_detector.main()
        self.assertGreaterEqual(calls[0],3)
        send.assert_not_called()
        power.request.assert_called()
        frames.close.assert_called_once()


if __name__=='__main__':unittest.main()
