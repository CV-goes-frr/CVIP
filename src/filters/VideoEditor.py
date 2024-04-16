from multiprocessing import Pool

import numpy as np

from .Filter import Filter
from .MotionTracking import MotionTracking



class VideoEditor(Filter):
    """
    Class for editing videos.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def apply(frames: np.ndarray, processes_limit: int, pool: Pool, filter: type, num_frames: int, width: int, height: int):
        """
        Applies a filter to a sequence of video frames.

        Args:
            frames (np.ndarray): Array of frames.
            processes_limit (int): Number of processes for parallelization.
            pool (Pool): Process pool.
            filter (type): Filter to apply to the frames.
            num_frames (int): Number of frames in the video.
            width (int): Width of the video.
            height (int): Height of the video.

        Returns:
            List: Edited frames.
        """
        print("Start VideoEditor")
        output = np.empty((num_frames, height, width, 3), np.uint8)  # Create array of frames

        if type(filter) is MotionTracking:
            for index in range(len(frames) - 1):
                n_frame = filter.apply(frames[index], frames[index + 1], processes_limit, pool)  # Use filter to frame
                output[index, :, :, :] = n_frame[0]  # Add edited frame to array

            output[len(frames) - 1, :, :, :] = frames[-1]  # last frame but without detection
        else:
            index = 0  # index of frame
            for frame in frames:
                frame = filter.apply(frame, processes_limit, pool)  # Use filter to frame
                output[index, :, :, :] = frame[0]  # Add edited frame to array
                index += 1  # Next frame

        return [output]  # array of frames
