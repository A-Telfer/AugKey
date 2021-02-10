"""Test augkey/randaug.py"""

import unittest
import math
import numpy as np
import cv2

from PIL import Image
from augkey import randaug as ra
from augkey import operations as ops

# pylint: disable=no-self-use


class TestRandAug(unittest.TestCase):
    """Test the RandAug class"""

    def __init__(self, *args, **kwargs):
        super(TestRandAug, self).__init__(*args, **kwargs)

        # Create a test image
        image = np.zeros((400, 400, 3), 'uint8')

        # Draw a little red circle
        self.image = cv2.circle(  # pylint: disable=no-member
            image,
            center=(100, 100),
            radius=10,
            color=(255, 0, 0),
            thickness=-1
        )

    def test_plan_augment(self):
        """Test building an augmentation plan using plan_augment."""
        operations = [ops.Identity()]
        magnitude_policy = ra.MagnitudePolicy(value=1, policy='constant')
        augmenter = ra.RandAugment(
            N=10,
            operations=operations,
            magnitude_policy=magnitude_policy
        )
        magnitude, augmentation, directions = augmenter.plan_augment()

        # verify magnitude and augmentation length
        assert magnitude == 1
        assert len(augmentation) == 10
        for direction, operation in zip(directions, augmentation):
            # verify direction values and operation type is identity
            assert direction in [-1, 1]
            assert isinstance(operation, ops.Identity)

    def test_apply_image(self):
        """Test applying multiple operations to an image."""
        magnitude = 1
        augmentation = [
            ops.TranslateX([0, 100]),
            ops.TranslateY([0, 100])
        ]
        directions = [-1, -1]
        augmenter = ra.RandAugment()
        image = augmenter.apply_image(
            Image.fromarray(self.image.copy()),
            magnitude,
            augmentation,
            directions
        )

        # the pixel at (200, 200) should now be the center of a red circle
        output = np.array(image)[200, 200]
        target = np.array([255, 0, 0])
        assert np.array_equal(output, target)

    def test_apply_keypoints(self):
        """Test applying multiple operations with apply_keyptions."""
        magnitude = 1

        # Translate (100, 100)
        augmentation = [
            ops.TranslateX([0, 100]),
            ops.TranslateY([0, 100])
        ]
        directions = [-1, -1]
        augmenter = ra.RandAugment()
        output = augmenter.apply_keypoints(
            np.array([[100, 100]]),
            magnitude,
            augmentation,
            directions
        )
        target = np.array([200, 200])
        assert np.array_equiv(output, target)

    def test_apply_inv_keypoints(self):
        """Test that apply_inv_keypoints can revert transformations by
        apply_keypoints.
        """
        magnitude = 0
        augmentation = [
            ops.Rotate([math.pi/4]),
            ops.TranslateY([50]),
            ops.ShearX([0.5])
        ]
        directions = np.array([-1, 1, -1])
        augmenter = ra.RandAugment()
        inputs = np.array([[100, 100]])
        output = augmenter.apply_keypoints(
            inputs,
            magnitude,
            augmentation,
            directions
        )
        # inputs should not be the same as transformed outputs
        assert not np.allclose(inputs, output)

        output = augmenter.apply_inv_keypoints(
            output,
            magnitude,
            augmentation,
            directions
        )

        # inputs should now be the same as output
        assert np.allclose(inputs, output)

    def test_apply_image_and_keypoints(self):
        """Test that image and keypoints change together."""

        keypoints = np.array([
            [100, 111],  # outside of circle
            [111, 100],  # outside of circle
            [89, 100],  # outside of circle
            [100, 89],  # outside of circle
            [100, 100],  # inside of circle
        ])

        # verify starting pixels at keypoints are expected colours
        keypoint_is_red = [0, 0, 0, 0, 1]
        for i in range(5):
            x, y = keypoints[i]  # pylint: disable=invalid-name
            target = [255, 0, 0] if keypoint_is_red[i] else [0, 0, 0]
            assert np.array_equal(self.image[y, x], target)

        # transform points
        magnitude = 0
        augmentation = [
            ops.TranslateY([50]),
            ops.ShearX([0.5]),
            ops.Rotate([math.pi/8]),
        ]
        directions = np.array([-1, 1, 1])
        augmenter = ra.RandAugment()

        # transform image
        image = Image.fromarray(self.image)
        image = augmenter.apply_image(
            image,
            magnitude,
            augmentation,
            directions
        )
        image = np.array(image)

        # transform keypoints
        keypoints = augmenter.apply_keypoints(
            keypoints,
            magnitude,
            augmentation,
            directions
        )

        # round keypoints to nearest int
        keypoints = np.round(keypoints).astype(int)
        for i in range(5):
            x, y = keypoints[i]  # pylint: disable=invalid-name
            target = [255, 0, 0] if keypoint_is_red[i] else [0, 0, 0]
            assert np.array_equal(image[y, x], target)
