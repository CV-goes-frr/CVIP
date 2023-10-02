import numpy as np
from typing import List

from .Filter import Filter


class Duplicate(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:
        return [img, img]
