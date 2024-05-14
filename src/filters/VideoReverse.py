import cv2
import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class VideoReverse(Filter):
    def __init__(self):
        super().__init__()
        self.log = "REVERSING A VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        return np.flipud(frames)
