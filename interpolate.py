##################################################
#                                                #
#   Interpolates between frames that have been   #
#   cleaned (with background reinstated)         #
#                                                #
##################################################

# This code has been adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf_hub_film_example.ipynb

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model = hub.load("https://tfhub.dev/google/film/1")

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_image(img_url: str):
    """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""
    image_data = tf.io.read_file(img_url)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F


"""A wrapper class for running a frame interpolation based on the FILM model on TFHub

Usage:
    interpolator = Interpolator()
    result_batch = interpolator(image_batch_0, image_batch_1, batch_dt)
    Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
    (B,H,W,C) layout, batch_dt is the sub-frame time in range [0..1], (B,) layout.
"""

def _pad_to_align(x, align):
    """Pads image batch x so width and height divide by align.

    Args:
        x: Image batch to align.
        align: Number to align to.

    Returns:
        1) An image padded so width % align == 0 and height % align == 0.
        2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
            to undo the padding.
    """
    # Input checking.
    assert np.ndim(x) == 4
    assert align > 0, 'align must be a positive number.'

    height, width = x.shape[-3:-1]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    bbox_to_pad = {
            'offset_height': height_to_pad // 2,
            'offset_width': width_to_pad // 2,
            'target_height': height + height_to_pad,
            'target_width': width + width_to_pad
    }
    padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
    bbox_to_crop = {
            'offset_height': height_to_pad // 2,
            'offset_width': width_to_pad // 2,
            'target_height': height,
            'target_width': width
    }
    return padded_x, bbox_to_crop


class Interpolator:
    """A class for generating interpolated frames between two input frames.

    Uses the Film model from TFHub
    """

    def __init__(self, align: int = 64) -> None:
        """Loads a saved model.

        Args:
            align: 'If >1, pad the input size so it divides with this before
                inference.'
        """
        self._model = hub.load("https://tfhub.dev/google/film/1")
        self._align = align

    def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Generates an interpolated frame between given two batches of frames.

        All inputs should be np.float32 datatype.

        Args:
            x0: First image batch. Dimensions: (batch_size, height, width, channels)
            x1: Second image batch. Dimensions: (batch_size, height, width, channels)
            dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

        Returns:
            The result with dimensions (batch_size, height, width, channels).
        """
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align(x0, self._align)
            x1, _ = _pad_to_align(x1, self._align)

        inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        image = result['image']

        if self._align is not None:
            image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image.numpy()




####################
# === NEW CODE === #
####################
interp = Interpolator()
def run_interp(frame1: np.ndarray, frame2: np.ndarray):
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    return interp(np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]

from settings import *

import cv2
import os

def raw_check(filename):
    return not filename.endswith("_s.png") and filename != SHEET_NAME and filename.endswith('.png') and 'interp' not in filename

ITERATIONS = 2 # This will run 2^ITERATIONS times, (4 iterations = 16 new images per input pair)
LOOP_INTERP = True
interp_folder = f"{CLEAN_PATH}"
def to_interpolate(filename):
    return True
    # return "srun" in filename
    # return "frun" in filename
    # return "brun" in filename

assert os.path.exists(interp_folder), f"The interp folder does not exist: '{interp_folder}'"

files = [f for f in os.listdir(interp_folder) if raw_check(f) and to_interpolate(f)]

assert len(files) > 1, f"Found {len(files)} files in interp_folder, expected >= 2"
print(f"Interpolating between: {files}")

all_imgs = [load_image(f"{interp_folder}/{f}") for f in files]

for pair_i in range(len(files) - (0 if LOOP_INTERP else 1)):
    curr_imgs = [ all_imgs[pair_i], all_imgs[pair_i + 1 if pair_i+1<len(files) else 0] ]
    for iter in range(ITERATIONS):
        new_imgs = [curr_imgs[0]]
        for i in range(len(curr_imgs) - 1):
            new_imgs.append(run_interp(curr_imgs[i], curr_imgs[i+1]))
            new_imgs.append(curr_imgs[i + 1])
        curr_imgs = new_imgs

    for i in range(1, len(curr_imgs) - 1):
        img = tf.cast(np.clip(curr_imgs[i], 0, 1) * _UINT8_MAX_F, dtype=np.uint8).numpy()
        converted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{interp_folder}/" + files[pair_i].replace(".png", f"-interp{i:02}.png"), converted)
