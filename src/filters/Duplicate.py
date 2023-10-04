from typing import List
from multiprocessing import Pool

import numpy as np

from .Filter import Filter


class Duplicate(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        return [img, img]
