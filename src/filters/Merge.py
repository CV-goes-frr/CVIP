from multiprocessing import Pool
from typing import List

import numpy as np

from .Filter import Filter


class Merge(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img1: np.ndarray, img2: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("DUPLICATE IN PROCESS...")

        if self.calls_counter > 1:
            self.cache = [img2]

        return [img2]
