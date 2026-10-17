import unittest
import io
from types import SimpleNamespace
from unittest.mock import Mock
from power_watcher import decode_nec,Frames,Debounce,notify_power,close_input,detector_command

def raw(values):
    pairs=[9000,-4500]
    for v in values:
        for i in range(8):pairs.extend([560,-1690 if (v>>i)&1 else -560])
    return pairs+[560]

class Tests(unittest.TestCase):
    def test_power_and_mute_are_distinct(self):
        self.assertEqual(decode_nec(raw([18,52,86,169])),(18,52,86,169))
        self.assertEqual(decode_nec(raw([18,52,119,136])),(18,52,119,136))
    def test_noise_and_repeats_rejected(self):
        for p in [[],[9000,-2250,560],raw([18,52,86,86]),raw([18,52,86,169])[:-1]]:
            self.assertIsNone(decode_nec(p))
    def test_timing_tolerance(self):
        self.assertIsNotNone(decode_nec([int(x*1.1) for x in raw([18,52,86,169])]))
    def test_mode2_stream(self):
        f=Frames()
        for p in raw([18,52,86,169]):self.assertIsNone(f.feed(abs(p)|(0x1000000 if p>0 else 0)))
        self.assertEqual(f.feed(0x3000000|12664),(18,52,86,169))
    def test_holding_does_not_retrigger(self):
        d=Debounce();self.assertTrue(d.accept(0))
        for i in range(1,100):self.assertFalse(d.accept(i*.1))
        self.assertTrue(d.accept(12))

    def test_subsequent_power_is_delivered_to_existing_process(self):
        child=SimpleNamespace(stdin=io.StringIO())
        self.assertTrue(notify_power(child))
        self.assertEqual(child.stdin.getvalue(),'power\n')

    def test_exited_process_pipe_is_handled_without_reusing_it(self):
        child=SimpleNamespace(stdin=Mock())
        child.stdin.write.side_effect=BrokenPipeError
        child.stdin.close.side_effect=BrokenPipeError
        self.assertFalse(notify_power(child))
        child.stdin.flush.assert_not_called()
        close_input(child)

    def test_default_session_cannot_send_mute(self):
        self.assertNotIn('--send-ir',detector_command())

    def test_live_session_uses_verified_transmitter_and_signal(self):
        command=detector_command(True,'/dev/lirc0','/example/ir/mute.txt')
        self.assertIn('--send-ir',command)
        self.assertIn('--tv-starts-unmuted',command)
        self.assertEqual(command[command.index('--ir-device')+1],'/dev/lirc0')
        self.assertEqual(command[command.index('--ir-signal')+1],'/example/ir/mute.txt')
        with self.assertRaises(ValueError):detector_command(True)

if __name__=='__main__':unittest.main()
