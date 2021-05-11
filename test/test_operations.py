"""Test rkaug/operations.py"""

import unittest
import math
import numpy as np

from augkey import operations as ops

# pylint: disable=no-self-use


class TestKeypointOperations(unittest.TestCase):
    """Test the operations that transform keypoints."""

    def test_identity_keypoints(self):
        """Basic test, identity operation should not change inputs."""
        operation = ops.Identity()
        inputs = np.arange(20).reshape(10, 2)
        assert np.array_equal(inputs, operation.transform_keypoints(inputs))

    def test_rotate_keypoints_90(self):
        """Rotate 90 Clockwise."""
        operation = ops.Rotate([np.pi/2])
        inputs = np.identity(2)
        outputs = operation.transform_keypoints(
            inputs, magnitude=0, direction=1)
        target = np.array([[0, -1], [1, 0]])
        assert np.allclose(target, outputs)

    def test_rotate_keypoints_45(self):
        """Rotate 45 Counter-Clockwise."""
        operation = ops.Rotate([np.pi/4])
        inputs = np.identity(2)
        outputs = operation.transform_keypoints(
            inputs, magnitude=0, direction=1)
        value = math.cos(math.pi/4)
        target = np.array([[value, -value], [value, value]])
        assert np.allclose(target, outputs)

    def test_shearx_keypoints(self):
        """Test ShearX 0.5"""
        operation = ops.ShearY([0.5])
        inputs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        outputs = operation.transform_keypoints(
            inputs, magnitude=0, direction=-1)
        target = np.array([
            [0, 0],
            [0, 1],
            [1, 0.5],
            [1, 1.5]
        ])
        assert np.allclose(outputs, target)

    def test_sheary_keypoints(self):
        """Test ShearY 0.5"""
        operation = ops.ShearX([0.5])
        inputs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        outputs = operation.transform_keypoints(
            inputs, magnitude=0, direction=-1)
        target = np.array([
            [0, 0],
            [0.5, 1],
            [1, 0],
            [1.5, 1]
        ])
        assert np.allclose(outputs, target)

    def test_translatex_keypoints(self):
        """Test TranslateX 10"""
        operation = ops.TranslateX([10])
        inputs = np.array([
            [0, 1],
            [1, 0]
        ])
        outputs = operation.transform_keypoints(inputs, 0, -1)
        target = inputs.copy()
        target[:, 0] += 10
        assert np.allclose(outputs, target)

    def test_translatey_keypoints(self):
        """Test TranslateY 10"""
        operation = ops.TranslateY([10])
        inputs = np.array([
            [0, 1],
            [1, 0]
        ])
        outputs = operation.transform_keypoints(inputs, 0, -1)
        target = inputs.copy()
        target[:, 1] += 10
        assert np.allclose(outputs, target)

    def test_translatey_with_visibilities(self):
        operation = ops.TranslateY([10])
        inputs = np.array([
            [0, 0, 2],
            [10, 10, 1],
            [5, 5, 1],
            [0, 5, 2],
            [20, 20, 2],
        ])

        image_shape = (20, 10, 3)
        outputs = operation.transform_keypoints(inputs, 0, -1, image_shape)
        visibilities = outputs[:, 2]
        target_visibilities = np.array([2, 0, 1, 2, 0])

        print(visibilities)
        print(target_visibilities)
        self.assertTrue(np.allclose(visibilities, target_visibilities))