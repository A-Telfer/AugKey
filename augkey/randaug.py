#!python

"""
============================
Random Keypoint Augmentation
============================

This module contains an implementation of RandAugment and its policies.

References
----------
.. [1] Cubuk, Ekin D., Barret Zoph, Jonathon Shlens, and Quoc V. Le.
  “RandAugment: Practical Automated Data Augmentation with a Reduced Search
  Space.” ArXiv:1909.13719 [Cs], November 13, 2019.
  http://arxiv.org/abs/1909.13719.

.. [2] Cubuk, Ekin D., Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc
  V. Le. “AutoAugment: Learning Augmentation Policies from Data.”
  ArXiv:1805.09501 [Cs, Stat], April 11, 2019. http://arxiv.org/abs/1805.09501.

MIT License © 2021 Andre Telfer
"""

import math
import numpy as np

from augkey import operations as ops


class MagnitudePolicy:
    """This class generates magnitude values from a given policy."""

    def __init__(self, value=10, policy='random'):
        """Generates new magnitudes using a given policy and value.

        Parameters
        ----------
        value: int, default=10
            Used by the policy to select the magnitude.
        policy: {'random', 'constant'}, default='random'
            The policy used to select the next magnitude value. A random policy
            will select a magnitude between [0, value]. A constant policy will
            always return the value.

        References
        ----------
        .. [1] Cubuk, Ekin D., Barret Zoph, Jonathon Shlens, and Quoc V. Le.
          “RandAugment: Practical Automated Data Augmentation with a Reduced
          Search Space.” ArXiv:1909.13719 [Cs], November 13, 2019.
          http://arxiv.org/abs/1909.13719.
        """
        assert policy in ['random', 'constant']

        self.policy = policy
        self.value = value

    def __call__(self):
        """Return the next magnitude value.

        Returns
        -------
        magnitude : int
        """
        if self.policy == 'random':
            return np.random.randint(0, self.value+1)
        elif self.policy == 'constant':
            return self.value

        raise 'Magnitude policy not recognized'


class RandAugment:
    """

    References
    ----------
    .. [1] Cubuk, Ekin D., Barret Zoph, Jonathon Shlens, and Quoc V. Le.
      “RandAugment: Practical Automated Data Augmentation with a Reduced
      Search Space.” ArXiv:1909.13719 [Cs], November 13, 2019.
      http://arxiv.org/abs/1909.13719.
    """

    def __init__(self, N=2, operations=None, magnitude_policy=None):
        """Create a new RandAugment instance to generate augmentations.

        Parameters
        ----------
        N : int
            How many operations to use in an augmentation.
        operations : list of Operation
            The operations that can be selected to build an augmentation.
        magnitude_policy : MagnitudePolicy or callable
            A policy for selecting the next magnitude. By default this will
            be set to a random policy with a max magnitude of 10.
        """
        if operations is None:
            # default augmentations from RandAugment [Cubuk et al., 2019]
            self.operations = {
                'idetntity':    ops.Identity(),
                'autocontrast': ops.AutoContrast(),
                'equalize':     ops.Equalize(),
                'rotate':       ops.Rotate(np.linspace(0, math.radians(30), 30)),
                'solarize':     ops.Solarize(np.linspace(0, 256, 30, dtype=int)),
                'posterize':    ops.Posterize(np.linspace(4, 8, 30, dtype=int)),
                'color':        ops.Color(np.linspace(0, 0.9, 30)),
                'contrast':     ops.Contrast(np.linspace(0, 0.9, 30)),
                'brightness':   ops.Brightness(np.linspace(0, 0.9, 30)),
                'sharpness':    ops.Sharpness(np.linspace(0, 0.9, 30)),
                'shear-x':      ops.ShearX(np.linspace(0, 0.3, 30)),
                'shear-y':      ops.ShearY(np.linspace(0, 0.3, 30)),
                'translate-x':  ops.TranslateX(np.linspace(0, 150, 30)),
                'translate-y':  ops.TranslateY(np.linspace(0, 150, 30))
            }
        else:
            self.operations = dict(enumerate(operations))

        if magnitude_policy is None:
            self.magnitude_policy = MagnitudePolicy()
        else:
            self.magnitude_policy = magnitude_policy

        self.augmentation_length = N

    def plan_augment(self):
        """Selects magnitude, operations, and directions for an augmentation.

        Returns:
            magnitude : int
            augmentation : list of int
            directions : list of int with values {-1, 1}
        """
        # Select the magnitude
        magnitude = self.magnitude_policy()

        # The augmentation is a dict of operations
        aug_ops = np.random.choice(
            list(self.operations.keys()),
            size=self.augmentation_length
        )

        # Certain operations will have multiple directions
        directions = np.random.choice(
            [-1, 1],
            size=self.augmentation_length
        )

        return (magnitude, aug_ops, directions)

    # pylint: disable=no-self-use
    def apply_image(self, image, magnitude, operation_keys, directions):
        """Apply augmentation plan generated by `plan_augment()` to image.

        Parameters
        ----------
        image : PIL.Image
        magnitude : int
        operation_indices : list of int 
        directions : list of in with values {-1, 1}

        Returns
        -------
        image : PIL.Image
        """
        for direction, operation_key in zip(directions, operation_keys):
            operation = self.operations[operation_key]
            image = operation.transform_image(image, magnitude, direction)
        return image

    def apply_keypoints(self, keypoints, magnitude, operation_keys,
                        directions, image_shape):
        """Apply augmentation plan generated by `plan_augment()` to keypoints.

        Parameters
        ----------
        keypoints : np.array
          Keypoints array with 2 columns for {x, y} coordinates.
        magnitude : int
        operation_keys : list of int 
        directions : list of in with values {-1, 1}

        Returns
        -------
        keypoints : np.array
        """
        for direction, operation_key in zip(directions, operation_keys):
            operation = self.operations[operation_key]
            keypoints = operation.transform_keypoints(
                keypoints, magnitude, direction, image_shape)
        return keypoints

    def apply_inv_keypoints(self, keypoints, magnitude, operation_keys,
                            directions, image_shape):
        """Apply inverse of augmentation to keypoints.

        Takes a set of transformed keypoints and applies transformations to
        them in the reverse order in opposite direction.

        Parameters
        ----------
        keypoints : np.array
          Keypoints array with 2 columns for {x, y} coordinates.
        magnitude : int
        operation_keys : list of int
        directions : list of in with values {-1, 1}

        Returns
        -------
        keypoints : np.array
        """

        # directions must be a numpy array to be multiplied by -1
        assert isinstance(directions, np.ndarray)

        return self.apply_keypoints(
            keypoints,
            magnitude,
            np.flip(operation_keys),  # Reverse the order of augmentations
            np.flip(directions)*-1,  # Reverse the direction of augmentations
            image_shape
        )
