"""Small geometry helpers for MediaPipe facial landmarks.

Distances are measured in pixels, NOT raw normalized image coordinates.
This matters because webcam frames are normally not square.
"""
from math import hypot

# Indices from Aravind's original implementation (MediaPipe Face Landmarker).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308


def pixel_xy(landmark, frame_width, frame_height):
    """Convert a normalized x/y landmark into floating-point pixel coordinates."""
    return landmark.x * frame_width, landmark.y * frame_height


def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, eye_indices, frame_width, frame_height):
    """EAR = sum(two lid distances) / (2 * eye width)."""
    pts = [pixel_xy(landmarks[i], frame_width, frame_height) for i in eye_indices]
    horizontal = distance(pts[0], pts[3])
    if horizontal <= 1e-9:
        return None
    return (distance(pts[1], pts[5]) + distance(pts[2], pts[4])) / (2 * horizontal)


def mouth_opening_ratio(landmarks, frame_width, frame_height):
    """Simple opening/width ratio (not a standard multi-point MAR)."""
    top = pixel_xy(landmarks[UPPER_LIP], frame_width, frame_height)
    bottom = pixel_xy(landmarks[LOWER_LIP], frame_width, frame_height)
    left = pixel_xy(landmarks[LEFT_MOUTH], frame_width, frame_height)
    right = pixel_xy(landmarks[RIGHT_MOUTH], frame_width, frame_height)
    mouth_width = distance(left, right)
    if mouth_width <= 1e-9:
        return None
    return distance(top, bottom) / mouth_width
