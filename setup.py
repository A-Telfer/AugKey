#!python
from setuptools import setup

install_requires = [
    "numpy>=1.10",
    "Pillow>=7.0"
]

tests_require = [
    "opencv-python>=3",
]

setup(
    name='augkey',
    version='1.0',
    description='RandAugment for Images and Keypoints.',
    author='Andre Telfer',
    author_email='andretelfer@cmail.carleton.ca',
    url='https://github.com/A-Telfer/AugKey',
    packages=['augkey'],
    keywords=['Image Augmentation', "RandAugment",
              "Random Augmentation", "Keypoint Augmentation", "Augmentation"],
    install_requires=install_requires,
    tests_require=tests_require
)
