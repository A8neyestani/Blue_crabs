"""Regression tests for empty detections, mask geometry and metric inputs."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from crab import load_depth, projected_dimensions, read_image, segment


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1.]])
        self.mask = np.zeros((50, 80), dtype=bool)
        self.mask[10:21, 10:31] = True
        self.depth = np.ones(self.mask.shape)

    def test_known_rectangle(self):
        result = projected_dimensions(self.mask, self.depth, self.k)
        self.assertAlmostEqual(result['pca_axis_1_mm'], 200.)
        self.assertAlmostEqual(result['pca_axis_2_mm'], 100.)

    def test_invalid_depth_returns_none(self):
        for value in [0, -1, np.nan, np.inf]:
            self.assertIsNone(projected_dimensions(self.mask, np.full(self.mask.shape, value), self.k))

    def test_shape_and_calibration_errors(self):
        with self.assertRaises(ValueError):
            projected_dimensions(self.mask, self.depth[:20], self.k)
        for k in [np.eye(2), np.zeros((3, 3)), self.k * np.nan]:
            with self.assertRaises(ValueError):
                projected_dimensions(self.mask, self.depth, k)

    def test_depth_units_and_preview_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.png'
            cv2.imwrite(str(path), np.full((20, 20), 1000, dtype=np.uint16))
            np.testing.assert_allclose(load_depth(path, 0.001), 1.)
            cv2.imwrite(str(path), np.full((20, 20), 100, dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_depth(path, 0.001)

    def test_missing_image(self):
        with self.assertRaises(ValueError):
            read_image('missing-image.jpg')

    def test_no_detections(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(masks=None, boxes=None)]
        self.assertEqual(segment(model, np.zeros((45, 80, 3), np.uint8)), [])

    def test_same_class_instances_stay_separate(self):
        masks = np.zeros((2, 50, 80), dtype=float)
        masks[0, 10:21, 10:31] = 1
        masks[1, 10:21, 50:71] = 1
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = masks
        box = SimpleNamespace(cls=np.array(0), conf=np.array(0.9))
        model = Mock()
        model.predict.return_value = [SimpleNamespace(masks=SimpleNamespace(data=tensor),
                                                      boxes=[box, box], names={0: 'Blue_crab_Female'})]
        instances = segment(model, np.zeros((50, 80, 3), np.uint8))
        self.assertEqual(len(instances), 2)
        for item in instances:
            self.assertAlmostEqual(projected_dimensions(item.mask, self.depth, self.k)['pca_axis_1_mm'], 200.)
        self.assertTrue(model.predict.call_args.kwargs['retina_masks'])


if __name__ == '__main__':
    unittest.main()
