import numpy as np
from multiprocessing import Pool

from .Filter import Filter
from .MotionTracking import MotionTracking


class VideoEditor(Filter):

    def __init__(self):
        super().__init__()

    @staticmethod
    def apply(frames: np.ndarray, processes_limit: int, pool: Pool, filter: type, num_frames: int, width: int, height: int):
        """
        Class for video
        :param frames: Array of frames
        :param processes_limit: we'll try to parallel it later
        :param pool: processes pool
        :param filter: filter that we use to our video
        :param num_frames: videos number of frames
        :param width: video's width
        :param height: video's height
        :return:
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
