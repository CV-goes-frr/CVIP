from multiprocessing import Pool
from typing import List

import numpy as np

from .Filter import Filter


class Crop(Filter):

    def __init__(self, x_1: int, y_1: int, x_2: int, y_2: int):
        super().__init__()
        self.x_1: int = int(x_1)
        self.y_1: int = int(y_1)
        self.x_2: int = int(x_2)
        self.y_2: int = int(y_2)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """

        print("CROP IN PROCESS...")
        if self.cache:
            print("USING CACHE...")
            return self.cache

        input_height, input_width, _ = img.shape
        if (self.x_1 > input_width or self.y_1 > input_height or self.x_2 > input_width or self.y_2 > input_height) or (
                self.x_1 >= self.x_2 or self.y_1 >= self.y_2) or (
                self.x_1 < 0 or self.x_2 < 0 or self.y_1 < 0 or self.y_2 < 0) or (
                type(self.x_1) != int or type(self.x_2) != int or type(self.y_1) != int or type(self.y_2) != int):
            raise Exception(
                "Wrong crop parameters: " + str(self.x_1) + ' ' + str(self.y_1) + ' ' + str(self.x_2) + ' ' + str(self.y_2))

        result = [img[self.y_1:self.y_2, self.x_1:self.x_2]]

        if self.calls_counter > 1:
            self.cache = result

        return result
