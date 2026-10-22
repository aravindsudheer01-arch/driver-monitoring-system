"""Deterministic behavioral tests: no webcam / MediaPipe required."""
import unittest

from monitor import FatigueMonitor


class MonitorTests(unittest.TestCase):
    def test_thresholds_follow_eye_observations(self):
        m = FatigueMonitor()

        # Two open-eye observations to arm blink detector
        self.assertEqual(m.update(0, 0.332, 0.008).level, "NORMAL")
        self.assertEqual(m.update(0.05, 0.332, 0.008).level, "NORMAL")

        # Eyes close
        self.assertEqual(m.update(0.1, 0.044, 0.001).level, "EYES CLOSED")

        # Eyes remain closed
        self.assertTrue(m.update(0.2, 0.20, 0.001).eye_closed)

        # Eyes reopen
        self.assertEqual(m.update(0.3, 0.332, 0.008).level, "NORMAL")

        # One complete blink should be counted
        self.assertEqual(m.blinks, 1)

    def test_eye_closure_alert_escalation_once_each(self):
        m = FatigueMonitor()
        m.update(0, .33, .008)
        m.update(.1, .044, .008)
        status = m.update(1.6, .044, .008)  # Gap > 0.5 => clock resets!
        self.assertEqual(status.level, "EYES CLOSED")
        # Feed real frames at 0.1 s intervals.
        sounds = []
        for step in range(17, 51):
            status = m.update(step / 10, .044, .008)
            if status.audible_event:
                sounds.append(status.audible_event)
        self.assertIn("DROWSINESS WARNING", sounds)
        self.assertIn("POSSIBLE MICROSLEEP", sounds)
        self.assertEqual(sounds.count("DROWSINESS WARNING"), 1)
        self.assertEqual(sounds.count("POSSIBLE MICROSLEEP"), 1)

    def test_missing_face_resets_duration_but_not_counts(self):
        m = FatigueMonitor()
        m.update(0, .044, .008)
        m.update(.1, .044, .008)
        s = m.update(.2)
        self.assertEqual(s.level, "NO FACE")
        self.assertEqual(s.eye_closed_seconds, 0)
        self.assertEqual(m.update(.3, .044, .008).eye_closed_seconds, 0)

    def test_mouth_open_hold_detects_once(self):
        m = FatigueMonitor()
        m.update(0, .33, .01)
        for i in range(1, 13):
            s = m.update(i / 10, .33, .741)
        self.assertEqual(s.level, "POSSIBLE YAWN")
        self.assertEqual(s.possible_yawns, 1)
        for i in range(13, 20):
            s = m.update(i / 10, .33, .741)
        self.assertEqual(s.possible_yawns, 1)
        self.assertEqual(m.update(2.0, .33, .008).level, "NORMAL")

    def test_rolling_perclos_is_time_weighted_and_excludes_missing_face(self):
        m = FatigueMonitor(perclos_window=10, min_valid_seconds=1)
        m.update(0, .044, .008)
        for i in range(1, 11):
            m.update(i / 10, .044, .008)
        for i in range(11, 21):
            s = m.update(i / 10, .33, .008)
        self.assertIsNotNone(s.perclos_percent)
        self.assertAlmostEqual(s.perclos_percent, 55, delta=6)  # one frame latency
        # Lose face for 1 s; missing seconds excluded from coverage.
        for i in range(21, 31):
            s = m.update(i / 10, None, None)
        self.assertAlmostEqual(s.valid_seconds, 2.0, delta=.11)
        self.assertAlmostEqual(s.perclos_percent, 55, delta=6)

    def test_perclos_rolls_out_old_data(self):
        m = FatigueMonitor(perclos_window=2, min_valid_seconds=1)
        m.update(0, .044, .008)
        for i in range(1, 41):
            s = m.update(i / 10, .33 if i > 10 else .044, .008)
        self.assertIsNotNone(s.perclos_percent)
        self.assertAlmostEqual(s.perclos_percent, 0, delta=0.1)

    def test_no_perclos_before_minimum_observation(self):
        m = FatigueMonitor(min_valid_seconds=8)
        m.update(0, .33, .008)
        self.assertIsNone(m.update(.1, .33, .008).perclos_percent)

    def test_nonmonotonic_timestamps_rejected(self):
        m = FatigueMonitor()
        m.update(0, .33, .008)
        with self.assertRaises(ValueError):
            m.update(0, .33, .008)

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            FatigueMonitor(ear_close=.3, ear_open=.2)


if __name__ == "__main__":
    unittest.main()
