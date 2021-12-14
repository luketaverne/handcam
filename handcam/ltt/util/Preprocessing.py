"""
Preprocessing package

"""
import threading

import numpy as np
from typing import Callable, Tuple, Any

from scipy.ndimage import rotate

from handcam.ltt.util.Utils import apply_alpha_matting


class DataAugmentation(object):
    """
    Add after a mini-batch generator
    """

    def __init__(
        self,
        image_generator: Any,
        rotations: bool = False,
        hand_generator: Any = False,  # function that overlays random hands onto a batch
        center_crop: Tuple = False,  # use (224,224) format
        random_crop: Tuple = False,  # use (224,224) format
        preprocessor_: Callable = False,
        batch_size: int = 32,
    ) -> None:
        self.image_generator = image_generator
        self.rotations = rotations
        self.hand_generator = hand_generator
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.preprocessor = preprocessor_
        self.batch_size = batch_size

        # Check input arguments
        if self.center_crop is True and type(self.center_crop) is not type(tuple()):
            raise TypeError("center_crop must either be a tuple, or false")

        if self.center_crop and len(self.center_crop) != 2:
            raise TypeError("center_crop must a tuple shaped like (?, ?)")

        if self.random_crop is True and type(self.random_crop) is not type(tuple()):
            raise TypeError("random_crop must either be a tuple, or false")

        if self.random_crop and len(self.random_crop) != 2:
            raise TypeError("random_crop must a tuple shaped like (?, ?)")

        if self.center_crop and self.random_crop:
            raise ValueError(
                "Cannot enable both random cropping and center cropping. Choose one."
            )
        # original image size
        self.input_im_shape = self.image_generator.image_shape

        # set the crop size
        self.crop_size = self.random_crop or self.center_crop

        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> (np.ndarray, np.ndarray, np.ndarray):
        with self.lock:
            x_rgb, x_depth, y = next(self.image_generator)
            if self.rotations:
                # x_rgb, x_depth = self.__random_batch_rotate__(x_rgb, x_depth)
                pass
            if self.hand_generator:
                x_rgb, x_depth = self.__add_hands__(x_rgb, x_depth)
                # pass
            if self.center_crop:
                x_rgb, x_depth = self.__center_crop_batch__(
                    x_rgb, x_depth, crop_size=self.crop_size
                )
            if self.random_crop:
                pass
            if self.preprocessor:
                x_rgb, x_depth = self.preprocessor(x_rgb, x_depth)

            return (
                np.concatenate(
                    (
                        np.asarray(x_rgb, dtype=np.float32),
                        np.asarray(x_depth, dtype=np.float32),
                    ),
                    axis=3,
                ),
                y,
            )

    def __add_hands__(
        self, x_rgb: np.ndarray, x_depth: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        hands = next(self.hand_generator)
        # print(type(hands))
        # print(type(self.hand_generator.__next__()))
        # print(type(x_rgb))

        for i in range(x_rgb.shape[0]):
            x_rgb[i] = apply_alpha_matting(
                hands[i, :, :, 0:3], x_rgb[i], hands[i, :, :, 3]
            )
            x_depth[i] = apply_alpha_matting(
                np.zeros(shape=x_depth[i].shape), x_depth[i], hands[i, :, :, 3]
            )

        return x_rgb, x_depth

    # def __random_batch_rotate__(self, x_rgb: np.ndarray, x_depth: np.ndarray) -> (np.ndarray, np.ndarray):
    #     # Assume 10 -> -95 degrees of rotation, as this is the typical range for our setup.
    #     angles = np.random.randint(-95,10,x_rgb.shape[0])
    #     for i in range(x_rgb.shape[0]):
    #         x_rgb[i] = rotate(x_rgb[i], angle=angles[i], reshape=False)
    #         x_depth[i] = rotate(x_depth[i], angle=angles[i], reshape=False)
    #
    #     return x_rgb, x_depth

    def __center_crop_batch__(
        self, x_rgb: np.ndarray, x_depth: np.ndarray, crop_size: Tuple
    ) -> (np.ndarray, np.ndarray):
        """
        Just center crop the image

        :param self:
        :param x: [batch_size, y_dim, x_dim, channels]
        :param crop_size: [y,x] desired size
        :return:
        """

        y_dim = x_rgb.shape[1]
        x_dim = x_rgb.shape[2]

        y_start = int(np.floor((y_dim - self.crop_size[0]) / 2))
        y_end = y_start + crop_size[0]

        x_start = int(np.floor((x_dim - crop_size[1]) / 2))
        x_end = x_start + crop_size[1]

        return (
            x_rgb[:, y_start:y_end, x_start:x_end, :],
            x_depth[:, y_start:y_end, x_start:x_end],
        )


def simple_crop_batch(im, crop_size):
    """
    Just center crop the image

    :param self:
    :param im: [batch_size, y_dim, x_dim, channels]
    :param crop_size: [y,x] desired size
    :return:
    """

    out_batch = np.zeros((im.shape[0], crop_size[0], crop_size[1], im.shape[3]))
    y_dim = im.shape[1]
    x_dim = im.shape[2]

    y_start = int(np.floor((y_dim - crop_size[0]) / 2))
    y_end = y_start + crop_size[0]

    x_start = int(np.floor((x_dim - crop_size[1]) / 2))
    x_end = x_start + crop_size[1]

    return im[:, y_start:y_end, x_start:x_end, :]


def simple_crop_im(im, crop_size):
    """
    Just center crop the image

    :param self:
    :param im: [batch_size, y_dim, x_dim, channels]
    :param crop_size: [y,x] desired size
    :return:
    """

    # out_batch = np.zeros((im.shape[0], crop_size[0], crop_size[1], im.shape[3]))
    y_dim = 480
    x_dim = 640

    y_start = int(np.floor((y_dim - crop_size[0]) / 2))
    y_end = y_start + crop_size[0]

    x_start = int(np.floor((x_dim - crop_size[1]) / 2))
    x_end = x_start + crop_size[1]

    return im[y_start:y_end, x_start:x_end, :]
