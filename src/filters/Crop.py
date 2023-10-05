from multiprocessing import Pool
from typing import List

import numpy as np

from .Filter import Filter


class Crop(Filter):

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        super().__init__()
        self.x1: int = int(x1)
        self.y1: int = int(y1)
        self.x2: int = int(x2)
        self.y2: int = int(y2)

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
        if (self.x1 > input_width or self.y1 > input_height or self.x2 > input_width or self.y2 > input_height) or (
                self.x1 >= self.x2 or self.y1 >= self.y2) or (
                self.x1 < 0 or self.x2 < 0 or self.y1 < 0 or self.y2 < 0) or (
                type(self.x1) != int or type(self.x2) != int or type(self.y1) != int or type(self.y2) != int):
            raise Exception(
                "Wrong crop parameters: " + str(self.x1) + ' ' + str(self.y1) + ' ' + str(self.x2) + ' ' + str(self.y2))

        result = [img[self.y1:self.y2, self.x1:self.x2]]

        if self.calls_counter > 1:
            self.cache = result

        return result
