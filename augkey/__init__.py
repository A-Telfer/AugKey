"""An implementation of RandAug with support for keypoints.

Examples
--------

Basic usage for images
```python
import augkey

image = ... # your PIL.Image
randaug = augkey.RandAugment()
plan = randaug.plan_augment()
augmented_image = randaug.apply_image(image, *plan)
```

Create your own operations
```python
import augkey
from augkey import operations as ops

# Create your new operation.
class Crop(ops.Operation):
    def transform_image(self, image, magnitude=0, direction=1):
        value = self.magnitude_range[magnitude]
        width, height = im.size
        im.crop((value, height - value, width - value, value))

operations = [
    Crop(np.arange(30)), # Pass the magnitude_range in as an argument.
    ops.Rotate(np.linspace(0, math.radians(30), 30)),
]

randaug = ra.RandAug(operations=operations)
```
"""
from . import operations

from .randaug import *
