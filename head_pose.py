"""Coarse 3D head-pose estimation from six MediaPipe face landmarks.

Educational approximation, NOT metrically calibrated or safety-qualified.
The simplified 3D face is not an anatomically personalized model; focal
length assumes roughly the image width. Always calibrate a neutral pose.

OpenCV camera convention: x right, y down, z forward (away from camera).
The image used for BOTH landmark detection and pose estimation must be the
same mirrored/non-mirrored image, with matching width and height.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

# Correspondence order must match FACE_MODEL exactly.
LANDMARK_IDS = (1, 152, 33, 263, 61, 291)  # nose, chin, eye corners, mouth corners

# Stylized face coordinates, in arbitrary consistent units.
# Facial-feature geometry and webcam camera intrinsics are approximate.
FACE_MODEL = np.array([
    (0.0,   0.0, -90.0),   # nose tip protrudes towards camera
    (0.0, 300.0,  20.0),  # chin
    (-175.0, -130.0, 0.0),# one outer eye corner
    (175.0, -130.0, 0.0), # other outer eye corner
    (-115.0, 125.0, -10.0),
    (115.0, 125.0, -10.0),
], dtype=np.float64)


@dataclass(frozen=True)
class PoseEstimate:
    pitch: float  # x rotation in degrees, approximate
    yaw: float    # y rotation in degrees, approximate
    roll: float   # z rotation in degrees, approximate
    rms_pixels: float  # reprojection residual, NOT actual orientation uncertainty
    nose_xy: tuple[int, int]
    axis_xy: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def camera_matrix(width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    focal = float(width)  # Approximation: better results need real camera calibration
    return np.array([[focal, 0, width / 2],
                     [0, focal, height / 2],
                     [0, 0, 1]], dtype=np.float64)


def estimate_pose(landmarks, width, height, max_rms_pixels=15.0):
    """Return PoseEstimate, or None when landmarks/fit are unusable.

    Uses solvePnP + Rodrigues + RQDecomp3x3. The result describes face
    orientation RELATIVE to a calibrated neutral orientation, not gaze.
    """
    try:
        if len(landmarks) <= max(LANDMARK_IDS):
            return None
        image_points = np.array([
            (landmarks[i].x * width, landmarks[i].y * height)
            for i in LANDMARK_IDS
        ], dtype=np.float64)
        if not np.isfinite(image_points).all():
            return None
        K = camera_matrix(width, height)
        ok, rotation_vec, translation_vec = cv2.solvePnP(
            FACE_MODEL, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        projected, _ = cv2.projectPoints(FACE_MODEL, rotation_vec, translation_vec, K, None)
        projected = projected.reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
        if not math.isfinite(error) or error > max_rms_pixels:
            return None

        R, _ = cv2.Rodrigues(rotation_vec)
        angles = cv2.RQDecomp3x3(R)[0]  # Euler angles x, y, z in degrees
        if not all(math.isfinite(float(a)) for a in angles):
            return None

        # Draw local axes anchored on the nose: +x right, +y down, +z into head.
        origin = FACE_MODEL[0]
        axes = np.array([origin + (120, 0, 0),
                         origin + (0, 120, 0),
                         origin + (0, 0, 120)], dtype=np.float64)
        tips, _ = cv2.projectPoints(axes, rotation_vec, translation_vec, K, None)
        tips = tips.reshape(-1, 2)
        nose = tuple(np.rint(projected[0]).astype(int))
        axis_points = tuple(tuple(np.rint(p).astype(int)) for p in tips)
        return PoseEstimate(float(angles[0]), float(angles[1]), float(angles[2]),
                            error, nose, axis_points)
    except (ValueError, cv2.error, IndexError, TypeError):
        return None


def draw_axes(frame, pose):
    """Draw x(red), y(green), z(blue) head-coordinate axes."""
    if pose is None:
        return
    for end, color in zip(pose.axis_xy, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(frame, pose.nose_xy, end, color, 2, cv2.LINE_AA)
    cv2.circle(frame, pose.nose_xy, 4, (255, 255, 255), -1)
