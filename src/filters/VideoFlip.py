import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class VideoFlip(Filter):
    def __init__(self, axis: str):
        super().__init__()
        self.axis = axis
        self.log = "FLIPPING A VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        if self.axis == 'horizontal':
            return np.flip(frames, axis=2)
        else:
            return np.fliplr(frames)  # axis=1
