from multiprocessing import Pool
from typing import List

import numpy as np

from src.filters.Filter import Filter


class Merge(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img1: np.ndarray, img2: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img1: np.ndarray of pixels
        :param img2: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """

        print("MERGE IN PROCESS...")
        if self.cache:
            print("USING CACHE...")
            return self.cache

        if self.calls_counter > 1:
            self.cache = [img2]

        return [img2]
