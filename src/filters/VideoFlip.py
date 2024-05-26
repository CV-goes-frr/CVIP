import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class VideoFlip(Filter):
    def __init__(self, axis: str):
        """
        Initializes the VideoFilter filter.

        Args:
            axis (str): how the media should be flipped (horizontal or vertical).

        Returns:
            axis (str): how the media should be flipped (horizontal or vertical).
        """
        super().__init__()
        self.axis = axis
        self.log = "FLIPPING A VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        """
        Flips input media.

        Args:
            frames (np.ndarray): NumPy array of frames.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.
        Returns:
            np.ndarray: Array containing the flipped media frames.
        """
        if self.axis == 'horizontal':
            return np.flip(frames, axis=2)
        else:
            return np.fliplr(frames)  # axis=1
