from multiprocessing import Pool

import numpy as np

from .panorama.PanoramicMerge import PanoramicMerge
from .Filter import Filter


class VideoToPanorama(Filter):
    def __init__(self):
        """
        Initializes the VideoToPanorama filter.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.log = "CREATING PANORAMA FROM VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        """
        Converts video to panorama.

        Args:
            frames (np.ndarray): NumPy array of frames.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.
        Returns:
            np.ndarray: Array containing the panorama frame.
        """
        step = 30
        result = frames[0]

        for frame_index in range(step, len(frames), step):
            result = PanoramicMerge.process(result, frames[frame_index])

        return result
