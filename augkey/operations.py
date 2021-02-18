#!python

"""
=======================
AUGMENTATION OPERATIONS
=======================

This module contains operations that can be used to augment images and keypoint
matrices.

MIT License © 2021 Andre Telfer
"""

import math
import numpy as np

from PIL import Image, ImageOps, ImageEnhance

# Used as the default fill value when applying transformations to images.
GREY = (128, 128, 128)


def add_temp_ones_column(func):
    """Convenience function that adds a column of ones to the keypoints for
    affine transform.
    """

    def wrapper(*args, **kwargs):
        if 'keypoints' in kwargs:
            keypoints = kwargs['keypoints']
            kwargs['keypoints'] = np.c_[keypoints, np.ones(keypoints.shape[0])]
        else:
            args = list(args)
            keypoints = args[1]
            args[1] = np.c_[keypoints, np.ones(keypoints.shape[0])]

        return func(*args, **kwargs)[:, :2]

    return wrapper


class Operation:
    """Base class for augmentation operations."""

    def __init__(self, magnitude_range=None):
        """Initialize the operation.

        Parameters
        ----------
        magnitude_range : np.array
            Required for operations that use magnitudes.
        """
        self.magnitude_range = magnitude_range

    def transform_image(self, image, magnitude=0, direction=1):
        """Apply the operation to an image.

        Parameters
        ----------
        image : PIL.Image
          The image to transform.
        magnitude : int
          The magnitude of the distortion. Only required by some operations.
        direction : {-1, 1}
          For opeartions that can be applied in either direction. Only required
          by some operations.

        Returns
        -------
        image : PIL.Image
        """
        #pylint: disable=unused-argument
        return image

    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        """Apply the operation to keypoints.

        Parameters
        ----------
        keypoints : np.array
          The keypoints to transform.
        magnitude : int
          The magnitude of the distortion. Only required by some operations.
        direction : {-1, 1}
          For operations that can be applied in either direction. Only required
          by some operations.
        image_shape : (width, height)
          For operations that require the image's dimensions. The shape 
          should be in PIL's (width, height) order rather than the standard 
          nd.array (height, width).
        Returns
        -------
        keypoints : np.array
        """
        #pylint: disable=unused-argument
        return keypoints


class Identity(Operation):
    """Identity"""


class AutoContrast(Operation):
    """AutoContrast"""

    def transform_image(self, image, magnitude=0, direction=1):
        return ImageOps.autocontrast(image)


class Equalize(Operation):
    """Equalize"""

    def transform_image(self, image, magnitude=0, direction=1):
        return ImageOps.equalize(image)


class Solarize(Operation):
    """Solarize"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude]
        return ImageOps.solarize(image, value)


class Color(Operation):
    """Color"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude]
        return ImageEnhance.Color(image).enhance(value)


class Posterize(Operation):
    """Posterize"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude]
        return ImageOps.posterize(image, value)


class Contrast(Operation):
    """Contrast"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = 1 + self.magnitude_range[magnitude] * direction
        return ImageEnhance.Contrast(image).enhance(value)


class Brightness(Operation):
    """Brightness"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = 1 + self.magnitude_range[magnitude] * direction
        return ImageEnhance.Brightness(image).enhance(value)


class Sharpness(Operation):
    """Sharpness"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = 1 + self.magnitude_range[magnitude] * direction
        return ImageEnhance.Sharpness(image).enhance(value)


class Rotate(Operation):
    """Rotate"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude] * direction
        value = (math.cos(value), -math.sin(value), 0,
                 math.sin(value), math.cos(value), 0)
        return image.transform(image.size, Image.AFFINE, value, fillcolor=GREY)

    @add_temp_ones_column
    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        value = self.magnitude_range[magnitude] * -direction
        value = np.array([
            [math.cos(value), -math.sin(value), 0],
            [math.sin(value), math.cos(value), 0],
            [0, 0, 1]
        ])
        return keypoints @ value.T


class ShearX(Operation):
    """ShearX"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude] * direction
        value = (1, value, 0, 0, 1, 0)
        return image.transform(image.size, Image.AFFINE, value, fillcolor=GREY)

    @add_temp_ones_column
    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        value = self.magnitude_range[magnitude] * -direction
        value = np.array([
            [1, value, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        return keypoints @ value.T


class ShearY(Operation):
    """ShearY"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude] * direction
        value = (1, 0, 0, value, 1, 0)
        return image.transform(image.size, Image.AFFINE, value, fillcolor=GREY)

    @add_temp_ones_column
    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        value = self.magnitude_range[magnitude] * -direction
        value = np.array([
            [1, 0, 0],
            [value, 1, 0],
            [0, 0, 1]
        ])
        return keypoints @ value.T


class TranslateX(Operation):
    """TranslateX"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude] * direction
        value = (1, 0, value, 0, 1, 0)
        return image.transform(image.size, Image.AFFINE, value, fillcolor=GREY)

    @add_temp_ones_column
    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        value = self.magnitude_range[magnitude] * -direction
        value = np.array([
            [1, 0, value],
            [0, 1, 0],
            [0, 0, 1]
        ])
        return keypoints @ value.T


class TranslateY(Operation):
    """TranslateY"""

    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude] * direction
        value = (1, 0, 0, 0, 1, value)
        return image.transform(image.size, Image.AFFINE, value, fillcolor=GREY)

    @add_temp_ones_column
    def transform_keypoints(
        self,
        keypoints,
        magnitude=0,
        direction=1,
        image_shape=None
    ):
        value = self.magnitude_range[magnitude] * -direction
        value = np.array([
            [1, 0, 0],
            [0, 1, value],
            [0, 0, 1]
        ])
        return keypoints @ value.T
