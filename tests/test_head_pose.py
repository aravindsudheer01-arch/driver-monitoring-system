"""Synthetic projection checks: tests the PnP mathematics, not real-camera accuracy."""
import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from head_pose import FACE_MODEL, LANDMARK_IDS, camera_matrix, estimate_pose


def synthetic_landmarks(pitch=0., yaw=0., roll=0., width=640, height=480):
    """Project a known 3D face into an artificial camera image."""
    # Construct Rz @ Ry @ Rx, matching common cv2.RQDecomp3x3 convention.
    p, y, r = np.deg2rad([pitch, yaw, roll])
    Rx = np.array([[1.,0.,0.],[0.,np.cos(p),-np.sin(p)],[0.,np.sin(p),np.cos(p)]])
    Ry = np.array([[np.cos(y),0.,np.sin(y)],[0.,1.,0.],[-np.sin(y),0.,np.cos(y)]])
    Rz = np.array([[np.cos(r),-np.sin(r),0.],[np.sin(r),np.cos(r),0.],[0.,0.,1.]])
    vec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    tvec = np.array([0.,0.,950.], dtype=np.float64)
    projected, _ = cv2.projectPoints(FACE_MODEL, vec, tvec, camera_matrix(width,height), None)
    pts = projected.reshape(-1,2)
    landmarks = [SimpleNamespace(x=0.5,y=0.5) for _ in range(478)]
    for idx, (x,y) in zip(LANDMARK_IDS,pts):
        landmarks[idx] = SimpleNamespace(x=float(x)/width,y=float(y)/height)
    return landmarks


class HeadPoseTests(unittest.TestCase):
    def test_identity(self):
        result = estimate_pose(synthetic_landmarks(), 640,480)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.pitch,0,delta=.5)
        self.assertAlmostEqual(result.yaw,0,delta=.5)
        self.assertAlmostEqual(result.roll,0,delta=.5)
        self.assertLess(result.rms_pixels, .01)

    def test_yaw_rotation(self):
        result = estimate_pose(synthetic_landmarks(yaw=24.), 640,480)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.yaw,24,delta=1.)

    def test_roll_rotation(self):
        result = estimate_pose(synthetic_landmarks(roll=-27.), 640,480)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.roll,-27,delta=1.)

    def test_pitch_rotation(self):
        result = estimate_pose(synthetic_landmarks(pitch=19.), 640,480)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.pitch,19,delta=1.)

    def test_missing_landmarks(self):
        self.assertIsNone(estimate_pose([],640,480))

    def test_invalid_points(self):
        landmarks = synthetic_landmarks()
        landmarks[1].x = float('nan')
        self.assertIsNone(estimate_pose(landmarks,640,480))

    def test_reprojection_rejection(self):
        landmarks = synthetic_landmarks()
        landmarks[152].x += .25  # corrupt a chin measurement
        self.assertIsNone(estimate_pose(landmarks, 640,480, max_rms_pixels=5.))


if __name__ == '__main__': unittest.main()
