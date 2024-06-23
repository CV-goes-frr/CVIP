import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class Flip(Filter):
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
        self.log = "FLIPPING IN PROCESS..."

    def apply(self, frame: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        """
        Flips input media.

        Args:
            frame (np.ndarray): NumPy array.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.
        Returns:
            np.ndarray: Array containing the flipped image.
        """
        if self.axis == 'horizontal':
            return [np.flip(frame, axis=1)]
        else:
            return [np.flip(frame, axis=0)]
