"""Pure-Python temporal decisions for head orientation and fatigue fusion.

No OpenCV dependency. Head turning does NOT prove sleep. The combined
'POSSIBLE HEAD SLUMP' indicator requires both sustained tilt/nod AND an
existing eye-closure fatigue warning. Pose alone never triggers this claim.
"""
from dataclasses import dataclass


def angular_difference(angle, reference):
    """Minimal signed angular difference in degrees; handles +/-180 wrap."""
    return (angle - reference + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class PoseStatus:
    calibrated: bool
    valid: bool
    pitch: float | None
    yaw: float | None
    roll: float | None
    label: str
    away_seconds: float
    tilt_seconds: float
    possible_slump: bool


class PoseMonitor:
    def __init__(self, *, yaw_threshold=30., pitch_threshold=25.,
                 roll_threshold=25., away_seconds=1.5, tilt_seconds=1.0,
                 max_frame_gap=0.5):
        if min(yaw_threshold, pitch_threshold, roll_threshold) <= 0:
            raise ValueError('Pose thresholds must be positive')
        if away_seconds <= 0 or tilt_seconds <= 0 or max_frame_gap <= 0:
            raise ValueError('Pose durations must be positive')
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.away_threshold_s = away_seconds
        self.tilt_threshold_s = tilt_seconds
        self.max_frame_gap = max_frame_gap
        self.reference = None
        self._last_t = None
        self._away_since = None
        self._tilt_since = None

    def calibrate(self, angles):
        """Use one observed neutral position; invoke on C key while upright."""
        if angles is None:
            return False
        self.reference = (angles.pitch, angles.yaw, angles.roll)
        self._away_since = None
        self._tilt_since = None
        self._last_t = None
        return True

    def update(self, now, angles, fatigue_level='NORMAL'):
        if self._last_t is not None and now <= self._last_t:
            raise ValueError('Pose timestamps must increase')
        stalled = self._last_t is not None and now - self._last_t > self.max_frame_gap
        self._last_t = now
        if stalled:
            self._away_since = None
            self._tilt_since = None

        if self.reference is None:
            return PoseStatus(False, angles is not None, None, None, None,
                              'PRESS C TO CALIBRATE', 0., 0., False)
        if angles is None:
            self._away_since = None
            self._tilt_since = None
            return PoseStatus(True, False, None, None, None,
                              'POSE UNAVAILABLE', 0., 0., False)

        rel_pitch = angular_difference(angles.pitch, self.reference[0])
        rel_yaw = angular_difference(angles.yaw, self.reference[1])
        rel_roll = angular_difference(angles.roll, self.reference[2])
        away = abs(rel_yaw) >= self.yaw_threshold
        tilt = (abs(rel_roll) >= self.roll_threshold or
                abs(rel_pitch) >= self.pitch_threshold)

        if away:
            if self._away_since is None:
                self._away_since = now
        else:
            self._away_since = None
        if tilt:
            if self._tilt_since is None:
                self._tilt_since = now
        else:
            self._tilt_since = None

        away_duration = now - self._away_since if away else 0.
        tilt_duration = now - self._tilt_since if tilt else 0.
        sustained_tilt = tilt and tilt_duration >= self.tilt_threshold_s
        possible_slump = (sustained_tilt and fatigue_level in
                          ('DROWSINESS WARNING', 'POSSIBLE MICROSLEEP'))
        if possible_slump:
            label = 'POSSIBLE HEAD SLUMP'
        elif sustained_tilt:
            label = 'SUSTAINED HEAD TILT'
        elif away and away_duration >= self.away_threshold_s:
            label = 'SUSTAINED HEAD TURN'
        elif tilt or away:
            label = 'HEAD MOVING'
        else:
            label = 'HEAD NEUTRAL'
        return PoseStatus(True, True, rel_pitch, rel_yaw, rel_roll,
                          label, away_duration, tilt_duration, possible_slump)
