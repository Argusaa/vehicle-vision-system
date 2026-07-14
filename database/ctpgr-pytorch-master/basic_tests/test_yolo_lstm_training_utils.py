import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_yolo_pose_gesture_model import calculate_class_weights, delayed_labels


class YoloLstmTrainingUtilsTests(unittest.TestCase):
    def test_zero_delay_keeps_labels_without_aliasing(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        shifted = delayed_labels(labels, 0)
        np.testing.assert_array_equal(shifted, labels)
        self.assertIsNot(shifted, labels)

    def test_positive_delay_pads_with_no_gesture(self):
        labels = np.array([1, 2, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(delayed_labels(labels, 2), [0, 0, 1, 2])

    def test_negative_delay_is_rejected(self):
        with self.assertRaises(ValueError):
            delayed_labels(np.array([1]), -1)

    def test_zero_power_produces_uniform_weights(self):
        counts = np.arange(1, 10)
        np.testing.assert_allclose(calculate_class_weights(counts, power=0), np.ones(9))

    def test_no_gesture_multiplier_increases_relative_weight(self):
        counts = np.full(9, 100)
        weights = calculate_class_weights(counts, power=0.5, no_gesture_multiplier=2)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
