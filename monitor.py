"""Pure temporal logic for an educational webcam fatigue-alert prototype.

No MediaPipe / OpenCV dependency: can be tested with fabricated observations.
All timestamps are monotonic seconds, supplied by the caller.
Not validated for driving safety decisions.
"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorStatus:
    face_valid: bool
    eye_closed: bool
    eye_closed_seconds: float
    mouth_open: bool
    mouth_open_seconds: float
    blinks: int
    possible_yawns: int
    perclos_percent: float | None
    valid_seconds: float
    level: str
    audible_event: str | None


class BlinkTracker:
    """Independent short-closure detector; NEVER used for fatigue alarms.

    The monitor has deliberately different thresholds and timing for a blink
    versus long eye closure. Consecutive open samples arm the detector; a
    quick dip and recovery counts once. Missing faces and frame stalls reset
    pending candidates, never incrementing a blink.
    """

    def __init__(self, *, close=0.26, reopen=0.29,
                 min_seconds=0.045, max_seconds=0.65,
                 min_open_frames=2, min_drop=0.05,
                 cooldown_seconds=0.20, max_frame_gap=0.5):
        if not (0 < close < reopen < 5):
            raise ValueError("Require 0 < blink_close < blink_open < 5")
        if not (0 <= min_seconds < max_seconds):
            raise ValueError("Require 0 <= min_seconds < max_seconds")
        if min_open_frames < 1 or min_drop < 0 or cooldown_seconds < 0 or max_frame_gap <= 0:
            raise ValueError("Invalid blink timing / quality constraints")
        self.close = close
        self.reopen = reopen
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.min_open_frames = min_open_frames
        self.min_drop = min_drop
        self.cooldown_seconds = cooldown_seconds
        self.max_frame_gap = max_frame_gap
        self.count = 0
        self._last_t = None
        self._last_count_t = None
        self._open_frames = 0
        self._open_baseline = None
        self._candidate_since = None

    def _reset_pending(self):
        self._open_frames = 0
        self._open_baseline = None
        self._candidate_since = None

    def update(self, now, ear):
        if self._last_t is not None and now <= self._last_t:
            raise ValueError("Frame times must be strictly increasing")
        stalled = self._last_t is not None and now - self._last_t > self.max_frame_gap
        self._last_t = now
        if ear is None or not (0 <= ear < 5) or stalled:
            self._reset_pending()
            # Even if a valid sample arrives after a camera stall, don't use
            # it as evidence that a blink occurred across the missing time.
            return self.count

        if self._candidate_since is not None:
            elapsed = now - self._candidate_since
            if ear >= self.reopen:
                if (self.min_seconds <= elapsed <= self.max_seconds and
                        (self._last_count_t is None or
                         now - self._last_count_t >= self.cooldown_seconds)):
                    self.count += 1
                    self._last_count_t = now
                self._candidate_since = None
                self._open_frames = 1
                self._open_baseline = ear
            elif elapsed > self.max_seconds:
                self._reset_pending()  # sustained eye closure is NOT a blink
            return self.count

        # Require two (by default) observed open frames before a candidate.
        if ear >= self.reopen:
            self._open_frames += 1
            self._open_baseline = (ear if self._open_baseline is None else
                                   0.8 * self._open_baseline + 0.2 * ear)
        elif (ear <= self.close and self._open_frames >= self.min_open_frames
              and self._open_baseline is not None
              and self._open_baseline - ear >= self.min_drop):
            self._candidate_since = now
            self._open_frames = 0
        else:
            # Middle-range reading is not a confirmed open frame, but can
            # occur during the approach to a blink. Do not invent a blink.
            self._open_frames = 0
        return self.count


class FatigueMonitor:
    def __init__(self, *, ear_close=0.18, ear_open=0.23,
                 mar_open=0.50, mar_close=0.35,
                 drowsy_seconds=1.5, microsleep_seconds=3.0,
                 yawn_seconds=0.8, perclos_window=30.0,
                 min_valid_seconds=8.0, max_frame_gap=0.5,
                 blink_close=0.26, blink_open=0.29):
        if not (0 < ear_close < ear_open):
            raise ValueError("Require 0 < ear_close < ear_open")
        if not (0 < mar_close < mar_open):
            raise ValueError("Require 0 < mar_close < mar_open")
        if not (0 < drowsy_seconds < microsleep_seconds):
            raise ValueError("Require 0 < drowsy_seconds < microsleep_seconds")
        if not (0 < yawn_seconds and 0 < min_valid_seconds <= perclos_window):
            raise ValueError("Invalid yawn seconds / PERCLOS window")
        if max_frame_gap <= 0:
            raise ValueError("max_frame_gap must be positive")
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.mar_open = mar_open
        self.mar_close = mar_close
        self.drowsy_seconds = drowsy_seconds
        self.microsleep_seconds = microsleep_seconds
        self.yawn_seconds = yawn_seconds
        self.perclos_window = perclos_window
        self.min_valid_seconds = min_valid_seconds
        self.max_frame_gap = max_frame_gap
        self._intervals = deque()  # (start, end, was_eye_closed), observed face only
        self._last_t = None
        self._last_valid = False
        self._last_closed = False
        self._eyes_closed = False
        self._closed_since = None
        self._mouth_open = False
        self._mouth_since = None
        self._yawn_counted = False
        self._blink_tracker = BlinkTracker(
            close=blink_close, reopen=blink_open, max_frame_gap=max_frame_gap
        )
        self.blinks = 0
        self.possible_yawns = 0
        self._last_level = "NO FACE"

    def _recent_coverage(self, now):
        cutoff = now - self.perclos_window
        while self._intervals and self._intervals[0][1] <= cutoff:
            self._intervals.popleft()
        valid = closed = 0.0
        for start, end, was_closed in self._intervals:
            length = max(0.0, end - max(start, cutoff))
            valid += length
            if was_closed:
                closed += length
        return valid, closed

    def update(self, now, ear=None, mouth_ratio=None):
        """Update one frame; pass None for both metrics if the face is missing."""
        if self._last_t is not None and now <= self._last_t:
            raise ValueError("Frame times must be strictly increasing")
        valid = (ear is not None and mouth_ratio is not None
                 and 0 <= ear < 5 and 0 <= mouth_ratio < 5)
        # Blink detection is independent; it does not change eye-closed
        # duration, fatigue state, PERCLOS-like measure, or audible warnings.
        self.blinks = self._blink_tracker.update(now, ear if valid else None)

        # Attribute elapsed time to the preceding valid observation only
        # when consecutive frames are reasonably close in time. Missing
        # faces and camera stalls are unobserved time, not closed-eye time.
        if (valid and self._last_valid and self._last_t is not None
                and now - self._last_t <= self.max_frame_gap):
            self._intervals.append((self._last_t, now, self._last_closed))

        if not valid:
            self._eyes_closed = False
            self._closed_since = None
            self._mouth_open = False
            self._mouth_since = None
            self._yawn_counted = False
            level = "NO FACE"
            closed_seconds = mouth_seconds = 0.0
        else:
            was_closed = self._eyes_closed
            if was_closed:
                if ear >= self.ear_open:
                    self._eyes_closed = False
            elif ear <= self.ear_close:
                self._eyes_closed = True

            if self._eyes_closed and not was_closed:
                self._closed_since = now
            elif not self._eyes_closed and was_closed:
                self._closed_since = None

            # On a large frame gap, don't claim eyes were closed continuously.
            if (self._eyes_closed and self._last_t is not None
                    and now - self._last_t > self.max_frame_gap):
                self._closed_since = now
            closed_seconds = (max(0., now - self._closed_since)
                              if self._eyes_closed else 0.0)

            if self._mouth_open:
                if mouth_ratio <= self.mar_close:
                    self._mouth_open = False
                    self._mouth_since = None
                    self._yawn_counted = False
            elif mouth_ratio >= self.mar_open:
                self._mouth_open = True
                self._mouth_since = now
                self._yawn_counted = False

            if (self._mouth_open and self._last_t is not None
                    and now - self._last_t > self.max_frame_gap):
                self._mouth_since = now
            mouth_seconds = (max(0., now - self._mouth_since)
                             if self._mouth_open else 0.0)
            if (self._mouth_open and mouth_seconds >= self.yawn_seconds
                    and not self._yawn_counted):
                self.possible_yawns += 1
                self._yawn_counted = True

            if self._eyes_closed and closed_seconds >= self.microsleep_seconds:
                level = "POSSIBLE MICROSLEEP"
            elif self._eyes_closed and closed_seconds >= self.drowsy_seconds:
                level = "DROWSINESS WARNING"
            elif self._eyes_closed:
                level = "EYES CLOSED"
            elif self._mouth_open and mouth_seconds >= self.yawn_seconds:
                level = "POSSIBLE YAWN"
            else:
                level = "NORMAL"

        audible_event = (level if level in ("DROWSINESS WARNING", "POSSIBLE MICROSLEEP")
                         and level != self._last_level else None)
        self._last_level = level
        self._last_t = now
        self._last_valid = valid
        self._last_closed = self._eyes_closed if valid else False
        observed, closed = self._recent_coverage(now)
        perclos = (100.0 * closed / observed
                   if observed >= self.min_valid_seconds and observed > 0 else None)
        return MonitorStatus(
            face_valid=valid,
            eye_closed=self._eyes_closed,
            eye_closed_seconds=closed_seconds,
            mouth_open=self._mouth_open,
            mouth_open_seconds=mouth_seconds,
            blinks=self.blinks,
            possible_yawns=self.possible_yawns,
            perclos_percent=perclos,
            valid_seconds=observed,
            level=level,
            audible_event=audible_event,
        )
