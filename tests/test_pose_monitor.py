"""Pose/time/fatigue fusion tested without a webcam or computer vision model."""
import unittest
from types import SimpleNamespace
from pose_monitor import PoseMonitor, angular_difference


def angles(pitch=0,yaw=0,roll=0):
    return SimpleNamespace(pitch=pitch,yaw=yaw,roll=roll)


class PoseMonitorTests(unittest.TestCase):
    def test_requires_user_calibration(self):
        m=PoseMonitor()
        self.assertEqual(m.update(0.,angles()).label,'PRESS C TO CALIBRATE')
        self.assertTrue(m.calibrate(angles(10,10,10)))
        self.assertEqual(m.update(.1,angles(10,10,10)).label,'HEAD NEUTRAL')

    def test_short_turn_is_not_sustained(self):
        m=PoseMonitor()
        m.calibrate(angles())
        self.assertEqual(m.update(0, angles(yaw=40)).label,'HEAD MOVING')
        self.assertEqual(m.update(.3, angles()).label,'HEAD NEUTRAL')

    def test_sustained_turn_is_visual_only(self):
        m=PoseMonitor()
        m.calibrate(angles())
        s=None
        for i in range(21):
            s=m.update(i*.1,angles(yaw=35),'NORMAL')
        self.assertEqual(s.label,'SUSTAINED HEAD TURN')
        self.assertFalse(s.possible_slump)

    def test_sustained_side_tilt_without_closed_eyes_is_not_sleep(self):
        m=PoseMonitor()
        m.calibrate(angles())
        s=None
        for i in range(16):
            s=m.update(i*.1,angles(roll=33),'NORMAL')
        self.assertEqual(s.label,'SUSTAINED HEAD TILT')
        self.assertFalse(s.possible_slump)

    def test_sustained_tilt_with_existing_fatigue_warning(self):
        m=PoseMonitor()
        m.calibrate(angles())
        for i in range(16):
            s=m.update(i*.1, angles(roll=32),
                       'DROWSINESS WARNING' if i>=15 else 'EYES CLOSED')
        self.assertTrue(s.possible_slump)
        self.assertEqual(s.label,'POSSIBLE HEAD SLUMP')

    def test_yaw_alone_with_closed_eyes_not_head_slump(self):
        m=PoseMonitor()
        m.calibrate(angles())
        for i in range(20):
            s=m.update(i*.1,angles(yaw=50),'POSSIBLE MICROSLEEP')
        self.assertFalse(s.possible_slump)

    def test_no_pose_resets_hold_timer(self):
        m=PoseMonitor()
        m.calibrate(angles())
        for i in range(8):
            m.update(i*.1, angles(roll=33))
        self.assertEqual(m.update(.8,None).label,'POSE UNAVAILABLE')
        self.assertEqual(m.update(.9,angles(roll=33)).tilt_seconds,0)

    def test_camera_frame_stall_resets_timer(self):
        m=PoseMonitor()
        m.calibrate(angles())
        m.update(0,angles(yaw=45))
        s=m.update(2,angles(yaw=45))
        self.assertEqual(s.away_seconds,0)
        self.assertEqual(s.label,'HEAD MOVING')

    def test_relative_baseline_and_wraparound(self):
        m=PoseMonitor()
        m.calibrate(angles(2,175,-175))
        s=m.update(1.,angles(4,-175,175))
        self.assertAlmostEqual(s.pitch,2)
        self.assertAlmostEqual(s.yaw,10)
        self.assertAlmostEqual(s.roll,-10)
        self.assertAlmostEqual(angular_difference(-179,179),2)

    def test_recovery_resets_slump(self):
        m=PoseMonitor()
        m.calibrate(angles())
        for i in range(16):
            m.update(i*.1,angles(roll=34),'DROWSINESS WARNING')
        s=m.update(1.6, angles(), 'NORMAL')
        self.assertEqual(s.label, 'HEAD NEUTRAL')
        self.assertFalse(s.possible_slump)

    def test_missing_face_never_confirms_slump(self):
        m=PoseMonitor()
        m.calibrate(angles())
        m.update(0,angles(roll=33),'DROWSINESS WARNING')
        s=m.update(.1,None,'POSSIBLE MICROSLEEP')
        self.assertFalse(s.possible_slump)

    def test_validate_thresholds(self):
        with self.assertRaises(ValueError): PoseMonitor(roll_threshold=0)


if __name__=='__main__': unittest.main()
