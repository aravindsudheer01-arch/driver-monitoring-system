"""Driver Monitoring System 

Run: python main.py --camera 1
C: neutral pose calibration; M: sound on/off; Q / Esc: quit.
Not validated for real-world driving safety decisions.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alerts import SoundAlerts
from head_pose import estimate_pose, draw_axes
from metrics import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, mouth_opening_ratio
from monitor import FatigueMonitor
from pose_monitor import PoseMonitor

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'


def draw_text(img, text, y, color=(230, 230, 230), scale=0.53):
    cv2.putText(img, str(text), (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_features(frame, landmarks):
    h, w = frame.shape[:2]
    for index in set(LEFT_EYE + RIGHT_EYE + (13, 14, 78, 308, 1, 152, 61, 291)):
        p = landmarks[index]
        cv2.circle(frame, (round(p.x * w), round(p.y * h)), 2,
                   (0, 255, 0), -1)


def status_panel(height, width=415):
    """Separate sidebar so measurements don't cover the camera image."""
    return np.full((height, width, 3), 25, dtype=np.uint8)


def run(args):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Missing face model: {MODEL_PATH}. Run python download_model.py')
    cap = cv2.VideoCapture(args.camera)
    if args.width > 0 and args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {args.camera}; try --camera 0 or 2')

    print('Camera resolution:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x',
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
    )
    fatigue = FatigueMonitor(
        ear_close=args.ear_close, ear_open=args.ear_open,
        mar_open=args.mar_open, mar_close=args.mar_close,
        blink_close=args.blink_close, blink_open=args.blink_open,
    )
    pose_monitor = PoseMonitor(yaw_threshold=args.yaw_threshold,
                               pitch_threshold=args.pitch_threshold,
                               roll_threshold=args.roll_threshold,
                               away_seconds=args.away_seconds,
                               tilt_seconds=args.tilt_seconds)
    sound = SoundAlerts(muted=args.mute)
    start = time.monotonic()
    prev_timestamp_ms = -1
    prev_frame_time = start
    smoothed_fps = None
    last_debug_t = start
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print('No camera frame available; stopping.')
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                timestamp_ms = max(prev_timestamp_ms + 1,
                                   int((time.monotonic() - start) * 1000))
                prev_timestamp_ms = timestamp_ms
                t0 = time.monotonic()
                result = detector.detect_for_video(image, timestamp_ms)
                inference_ms = 1000 * (time.monotonic() - t0)
                now = time.monotonic()
                dt = now - prev_frame_time
                prev_frame_time = now
                if dt > 0:
                    fps = 1. / dt
                    smoothed_fps = fps if smoothed_fps is None else .85 * smoothed_fps + .15 * fps

                ear = mouth = left_ear = right_ear = None
                pose = None
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    draw_features(frame, landmarks)
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                    if left_ear is not None and right_ear is not None:
                        ear = (left_ear + right_ear) / 2
                    mouth = mouth_opening_ratio(landmarks, w, h)
                    pose = estimate_pose(landmarks, w, h,
                                         max_rms_pixels=args.max_reprojection_error)
                    draw_axes(frame, pose)

                fatigue_status = fatigue.update(now, ear, mouth)
                pose_status = pose_monitor.update(now, pose, fatigue_status.level)
                # Keep the proven Phase 2.1 audio policy: eye-closure warning.
                # Sustained head turn alone DOES NOT trigger the fatigue alarm.
                sound.update(fatigue_status.level in ('DROWSINESS WARNING',
                                                      'POSSIBLE MICROSLEEP'))
                if args.debug_pose and now - last_debug_t >= 1.0:
                    print(f'pose raw={None if pose is None else (round(pose.pitch,1),round(pose.yaw,1),round(pose.roll,1))} '
                          f'rel={(pose_status.pitch,pose_status.yaw,pose_status.roll)} '
                          f'reproj={None if pose is None else round(pose.rms_pixels,1)} '
                          f'fps={smoothed_fps:.1f}' if smoothed_fps is not None else 'FPS unavailable')
                    last_debug_t = now

                panel = status_panel(h)
                green = (80, 240, 80)
                orange = (0, 170, 255)
                red = (70, 70, 255)
                draw_text(panel, 'DRIVER MONITORING  |  PHASE 3', 25, (0, 220, 255), .63)
                draw_text(panel, f'Face: {"detected" if fatigue_status.face_valid else "NOT DETECTED"}',
                          53, green if fatigue_status.face_valid else orange)
                draw_text(panel, f'EAR: {ear:.3f}' if ear is not None else 'EAR: N/A', 79)
                draw_text(panel, f'Mouth ratio: {mouth:.3f}' if mouth is not None else 'Mouth ratio: N/A', 103)
                draw_text(panel, f'Eyes closed: {fatigue_status.eye_closed_seconds:.1f} s', 127)
                draw_text(panel, f'Blinks: {fatigue_status.blinks} | yawns: {fatigue_status.possible_yawns}', 151)
                pct = (f'{fatigue_status.perclos_percent:.1f}%'
                       if fatigue_status.perclos_percent is not None else 'collecting')
                draw_text(panel, f'PERCLOS-like 30s: {pct}', 175)
                cv2.line(panel, (14, 193), (405, 193), (80, 80, 80), 1)
                if pose_status.valid and pose_status.calibrated:
                    draw_text(panel, f'Pitch: {pose_status.pitch:+.1f} deg  '
                              f'Yaw: {pose_status.yaw:+.1f} deg', 218)
                    draw_text(panel, f'Roll: {pose_status.roll:+.1f} deg  '
                              f'Reproj: {pose.rms_pixels:.1f} px', 242)
                elif pose_status.calibrated:
                    draw_text(panel, 'Head pose: unavailable', 218, orange)
                else:
                    draw_text(panel, 'Head pose: press C while facing forward', 218, orange, .47)
                draw_text(panel, f'HEAD: {pose_status.label}', 269,
                          orange if pose_status.label != 'HEAD NEUTRAL' else green, .50)
                draw_text(panel, f'Turn: {pose_status.away_seconds:.1f}s  '
                          f'Tilt: {pose_status.tilt_seconds:.1f}s', 293)
                cv2.line(panel, (14, 308), (405, 308), (80, 80, 80), 1)
                draw_text(panel, f'EYES: {fatigue_status.level}', 333,
                          red if fatigue_status.level in ('DROWSINESS WARNING',
                                                           'POSSIBLE MICROSLEEP') else green,
                          .52)
                if pose_status.possible_slump:
                    draw_text(panel, 'COMBINED: POSSIBLE HEAD SLUMP', 357, red, .51)
                else:
                    draw_text(panel, 'COMBINED: no added slump evidence', 357)
                draw_text(panel, f'Sound: {"OFF" if sound.muted else "ON"}  (M to toggle)', 380)
                draw_text(panel, 'C: calibrate upright | Q/Esc: exit', 404)
                if h >= 460:
                    draw_text(panel, f'FPS: {smoothed_fps:.1f} | model: {inference_ms:.0f}ms'
                              if smoothed_fps is not None else 'FPS: N/A', 428)
                    draw_text(panel, 'EDUCATIONAL DEMO - NOT FOR DRIVING', 455, orange, .44)
                cv2.imshow('Driver Monitoring System - Phase 3', np.hstack((frame, panel)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('c'):
                    if pose is not None and fatigue_status.face_valid:
                        pose_monitor.calibrate(pose)
                        print('Neutral head pose calibrated. Keep camera fixed.')
                    else:
                        print('Calibration unavailable: face/pose not tracked.')
                if key == ord('m'):
                    sound.muted = not sound.muted
                    if sound.muted:
                        sound.stop()
    finally:
        sound.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Educational fatigue + head pose demo')
    parser.add_argument('--camera', type=int, default=1,
                        help='Camera index; Aravind MacBook worked at 1')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--ear-close', type=float, default=.18)
    parser.add_argument('--ear-open', type=float, default=.23)
    parser.add_argument('--mar-open', type=float, default=.50)
    parser.add_argument('--mar-close', type=float, default=.35)
    parser.add_argument('--blink-close', type=float, default=.26)
    parser.add_argument('--blink-open', type=float, default=.29)
    parser.add_argument('--yaw-threshold', type=float, default=30.)
    parser.add_argument('--pitch-threshold', type=float, default=25.)
    parser.add_argument('--roll-threshold', type=float, default=25.)
    parser.add_argument('--away-seconds', type=float, default=1.5)
    parser.add_argument('--tilt-seconds', type=float, default=1.)
    parser.add_argument('--max-reprojection-error', type=float, default=15.)
    parser.add_argument('--mute', action='store_true')
    parser.add_argument('--debug-pose', action='store_true')
    run(parser.parse_args())
