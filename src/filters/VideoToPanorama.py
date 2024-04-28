from multiprocessing import Pool

import numpy as np

from .panorama.PanoramicMerge import PanoramicMerge
from .Filter import Filter


class VideoToPanorama(Filter):
    def __init__(self):
        super().__init__()
        self.log = "CREATING PANORAMA FROM VIDEO IN PROCESS..."

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        # print(frames.shape)
        step = 30
        result = frames[0]

        for frame_index in range(step, len(frames), step):
            result = PanoramicMerge.process(result, frames[frame_index])

        return result
