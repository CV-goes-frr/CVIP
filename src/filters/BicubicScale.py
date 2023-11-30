from functools import lru_cache
from multiprocessing import Pool
import math
from typing import List

import numpy as np

from .Filter import Filter
from .decorators.bicubic_hermit_decorator import bicubic_hermit_cache


class BicubicScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    @staticmethod
    def process_pixel(x: int, y: int, scale_factor: float,
                      input_width: int, input_height: int, img: np.ndarray) -> np.ndarray:
        """
        Method to parallel computations (it's static because we need to call it in other processes).
        Method gets 16 nearest pixels to make computations with bicubic hermit to set color of the pixel at (x,y).

        :param x: x position of pixel that's going to be processed
        :param y: y position of pixel that's going to be processed
        :param scale_factor: float value that we use to scale image
        :param input_width: width of the input image
        :param input_height: height of the input image
        :param img: np.ndarray of image pixels (2D)
        :return: (R, G, B) np.ndarray that is our processed pixel
        """
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)

        dx = original_x - math.floor(original_x)
        dy = original_y - math.floor(original_y)

        x_1 = min(max(math.floor(original_x) - 1, 0), input_width - 1)
        x_2 = min(max(math.floor(original_x), 0), input_width - 1)
        x_3 = min(max(math.floor(original_x) + 1, 0), input_width - 1)
        x_4 = min(max(math.floor(original_x) + 2, 0), input_width - 1)

        y_1 = min(max(math.floor(original_y) - 1, 0), input_height - 1)
        y_2 = min(max(math.floor(original_y), 0), input_height - 1)
        y_3 = min(max(math.floor(original_y) + 1, 0), input_height - 1)
        y_4 = min(max(math.floor(original_y) + 2, 0), input_height - 1)

        pix11 = img[y_1, x_1]
        pix21 = img[y_1, x_2]
        pix31 = img[y_1, x_3]
        pix41 = img[y_1, x_4]

        pix12 = img[y_2, x_1]
        pix22 = img[y_2, x_2]
        pix32 = img[y_2, x_3]
        pix42 = img[y_2, x_4]

        pix13 = img[y_3, x_1]
        pix23 = img[y_3, x_2]
        pix33 = img[y_3, x_3]
        pix43 = img[y_3, x_4]

        pix14 = img[y_4, x_1]
        pix24 = img[y_4, x_2]
        pix34 = img[y_4, x_3]
        pix44 = img[y_4, x_4]

        arr1 = bicubic_hermit(pix11, pix21, pix31, pix41, dy)
        arr2 = bicubic_hermit(pix12, pix22, pix32, pix42, dy)
        arr3 = bicubic_hermit(pix13, pix23, pix33, pix43, dy)
        arr4 = bicubic_hermit(pix14, pix24, pix34, pix44, dy)

        val = bicubic_hermit(arr1, arr2, arr3, arr4, dx)

        return val.astype(np.uint8)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """

        print("BICUBIC SCALE IN PROGRESS...")
        if self.cache:
            print("USING CACHE...")
            return self.cache

        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        part_height = new_height // processes_limit
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]
        parts = [coordinates[i:i + part_height] for i in range(0, len(coordinates), part_height)]

        processed_pixels = pool.starmap(self.process_pixel,
                                        [(x, y, self.scale_factor, input_width, input_height, img)
                                         for part in parts for (x, y) in part])

        for (x, y), pixel_value in zip(coordinates, processed_pixels):
            upscaled_image[y, x] = pixel_value

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]


@bicubic_hermit_cache
def bicubic_hermit(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t):
    a_n = -1 * a / 2 + 3 * b / 2 - 3 * c / 2 + d / 2
    b_n = a - 5 * b / 2 + 2 * c - d / 2
    c_n = -1 * a / 2 + c / 2
    d_n = b

    return a_n * pow3(t) + b_n * t * t + c_n * t + d_n


@lru_cache(maxsize=128)
def pow3(t: int):
    return t * t * t
