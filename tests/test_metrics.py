"""Geometry unit tests (do not require OpenCV, a webcam, or the model)."""
import unittest
from types import SimpleNamespace

from metrics import (
    LEFT_EYE, eye_aspect_ratio, mouth_opening_ratio, pixel_xy,
)


class MetricsTests(unittest.TestCase):
    def test_pixel_conversion_respects_aspect_ratio(self):
        pt = SimpleNamespace(x=0.5, y=0.5)
        self.assertEqual(pixel_xy(pt, 640, 480), (320, 240))

    def test_eye_aspect_ratio(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
        # Eye horizontal width 40 px; both vertical lid distances 20 px.
        coords = ((10, 20), (20, 10), (40, 10),
                  (50, 20), (40, 30), (20, 30))
        for index, (x, y) in zip(LEFT_EYE, coords):
            landmarks[index] = SimpleNamespace(x=x / 100, y=y / 100)
        self.assertAlmostEqual(eye_aspect_ratio(landmarks, LEFT_EYE, 100, 100), 0.5)

    def test_zero_eye_width_is_undefined(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
        self.assertIsNone(eye_aspect_ratio(landmarks, LEFT_EYE, 640, 480))

    def test_mouth_ratio(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
        landmarks[13] = SimpleNamespace(x=0.5, y=0.4)
        landmarks[14] = SimpleNamespace(x=0.5, y=0.6)
        landmarks[78] = SimpleNamespace(x=0.3, y=0.5)
        landmarks[308] = SimpleNamespace(x=0.7, y=0.5)
        self.assertAlmostEqual(mouth_opening_ratio(landmarks, 100, 100), 0.5)


if __name__ == "__main__":
    unittest.main()
