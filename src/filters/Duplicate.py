from typing import List
from multiprocessing import Pool

import numpy as np

from .Filter import Filter


class Duplicate(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """
        print("DUPLICATE IN PROCESS...")
        return [img, img]
