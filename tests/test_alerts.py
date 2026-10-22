"""Small tests of audio-activation policy, without playing sounds."""
import unittest
from unittest.mock import patch
from alerts import SoundAlerts


class AlertTests(unittest.TestCase):
    def test_alarm_generated_as_wav(self):
        alarm = SoundAlerts()
        try:
            self.assertTrue(alarm._wav_path.exists())
            self.assertGreater(alarm._wav_path.stat().st_size, 1000)
        finally:
            alarm.close()

    def test_update_only_play_when_warning_active(self):
        alarm = SoundAlerts()
        try:
            with patch.object(alarm, 'play') as play, patch.object(alarm, 'stop') as stop:
                alarm.update(True)
                play.assert_called_once()
                stop.assert_not_called()
                alarm.update(False)
                play.assert_called_once()
        finally:
            alarm.close()

    def test_muted_warning_is_silent(self):
        alarm = SoundAlerts(muted=True)
        try:
            with patch.object(alarm, 'play') as play:
                alarm.update(True)
                play.assert_not_called()
        finally:
            alarm.close()

    def test_mac_cooldown_avoids_restarts(self):
        alarm = SoundAlerts(cooldown=2.2)
        try:
            with patch('alerts.sys.platform', 'darwin'), \
                 patch('alerts.subprocess.Popen') as popen, \
                 patch('alerts.time.monotonic', side_effect=[10.0, 10.3, 12.5]):
                proc = popen.return_value
                proc.poll.return_value = 0
                alarm.play()
                alarm.play()
                alarm.play()
                self.assertEqual(popen.call_count, 2)
        finally:
            alarm.close()


if __name__ == '__main__':
    unittest.main()
