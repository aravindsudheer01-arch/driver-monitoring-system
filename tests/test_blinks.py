"""Phase 2.1: short-event sensitivity without changing fatigue alarms."""
import unittest
from monitor import BlinkTracker, FatigueMonitor


class BlinkTrackerTests(unittest.TestCase):
    def test_one_sample_blink_at_12fps(self):
        b = BlinkTracker()
        self.assertEqual(b.update(0.0, .33), 0)
        self.assertEqual(b.update(.083, .33), 0)
        self.assertEqual(b.update(.166, .24), 0)  # not below fatigue .18!
        self.assertEqual(b.update(.249, .33), 1)

    def test_partial_blink_no_fatigue_alert(self):
        m = FatigueMonitor()
        m.update(0, .33, .008)
        m.update(.083, .33, .008)
        s = m.update(.166, .24, .008)
        self.assertEqual(s.level, "NORMAL")
        s = m.update(.249, .33, .008)
        self.assertEqual(s.blinks, 1)
        self.assertEqual(s.level, "NORMAL")
        self.assertIsNone(s.audible_event)

    def test_flat_open_eye_no_false_blinks(self):
        b = BlinkTracker()
        for i in range(100):
            result = b.update(i*.083, .33)
        self.assertEqual(result, 0)

    def test_small_lighting_jitter_no_false_blinks(self):
        b = BlinkTracker()
        for i in range(100):
            result = b.update(i*.083, [.33,.31,.285,.32,.30][i % 5])
        self.assertEqual(result, 0)

    def test_sustained_closure_is_not_blink(self):
        b = BlinkTracker()
        b.update(0, .33)
        b.update(.083, .33)
        for i in range(2, 15):
            b.update(i*.083, .08)
        self.assertEqual(b.update(15*.083, .33), 0)

    def test_no_face_does_not_create_blink(self):
        b = BlinkTracker()
        b.update(0, .33)
        b.update(.083, .33)
        b.update(.166, .2)
        b.update(.249, None)
        self.assertEqual(b.update(.332, .33), 0)

    def test_camera_stall_does_not_create_blink(self):
        b = BlinkTracker()
        b.update(0, .33)
        b.update(.083, .33)
        b.update(.166, .2)
        self.assertEqual(b.update(1.1, .33), 0)

    def test_refractory_prevents_double_count(self):
        b = BlinkTracker(min_open_frames=1)
        for t,e in [(0,.33),(.1,.2),(.2,.33),(.24,.2),(.3,.33)]:
            b.update(t,e)
        self.assertEqual(b.count, 1)

    def test_reject_small_drop_even_when_below_threshold(self):
        b = BlinkTracker(min_open_frames=1)
        b.update(0, .29)
        b.update(.1, .25)  # drop is only .04 (< .05)
        self.assertEqual(b.update(.2, .32), 0)

    def test_tunable_thresholds(self):
        b = BlinkTracker(close=.25, reopen=.30)
        b.update(0, .35)
        b.update(.08, .35)
        b.update(.16, .24)
        self.assertEqual(b.update(.24, .35), 1)

    def test_still_warns_for_sustained_deep_eye_closure(self):
        m = FatigueMonitor()
        m.update(0, .33, .008)
        m.update(.1, .33, .008)
        for i in range(2, 41):
            s = m.update(i * .1, .04, .008)
        self.assertEqual(s.level, "POSSIBLE MICROSLEEP")
        self.assertEqual(s.blinks, 0)


if __name__ == "__main__":
    unittest.main()
