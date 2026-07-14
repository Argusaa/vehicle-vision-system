import unittest

import numpy as np

from app.services.police_gesture_service import PoliceGestureService


class PoliceGestureGateTests(unittest.TestCase):
    def test_keeps_confident_gesture(self):
        scores = np.array([0.0, 3.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.assertEqual(PoliceGestureService._apply_gesture_gate(scores, 0.35, 0.03), 1)

    def test_rejects_low_confidence_gesture(self):
        scores = np.array([0.0, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
        self.assertEqual(PoliceGestureService._apply_gesture_gate(scores, 0.35, 0.03), 0)

    def test_rejects_small_winner_margin(self):
        scores = np.array([0.0, 3.0, 2.95, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.assertEqual(PoliceGestureService._apply_gesture_gate(scores, 0.35, 0.03), 0)

    def test_never_rejects_no_gesture(self):
        scores = np.array([2.0, 1.9, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.assertEqual(PoliceGestureService._apply_gesture_gate(scores, 0.99, 0.99), 0)


if __name__ == "__main__":
    unittest.main()
