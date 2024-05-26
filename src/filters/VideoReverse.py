import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class VideoReverse(Filter):
    def __init__(self):
        """
        Initializes the VideoReverse filter.

        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.log = "REVERSING A VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        """
        Reverses input video.

        Args:
            frames (np.ndarray): NumPy array of frames.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.
        Returns:
            np.ndarray: Array containing the reversed video frames.
        """
        return np.flipud(frames)  # axis=0