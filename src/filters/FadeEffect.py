import cv2
import numpy as np
from multiprocessing import Pool

from .Filter import Filter


class FadeEffect(Filter):
    def __init__(self, fade_in_length: str, fade_out_length: str):
        """
        Initializes the FaceDetection filter.

        Args:
            fade_in_length (str): amount of frames that should be faded in the start of the video.
            fade_out_length (str): amount of frames that should be faded out the end of the video.

        Returns:
            fade_in_length (int): amount of frames that should be faded in the start of the video.
            fade_out_length (int): amount of frames that should be faded out the end of the video.
        """
        super().__init__()
        self.log = "APPLYING FADE FILTER TO VIDEO..."
        self.fade_in_length = int(fade_in_length)
        self.fade_out_length = int(fade_out_length)

    def apply(self, frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
        """
        Applies feature matching to the input images.

        Args:
            frames (np.ndarray): NumPy array of frames.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.

        Returns:
            List[np.ndarray]: List containing the edited video frames as a NumPy array.
        """
        num_frames = frames.shape[0]
        print(num_frames)

        # Ensure fade lengths do not exceed the number of frames
        fade_in_length = min(self.fade_in_length, num_frames)
        fade_out_length = min(self.fade_out_length, num_frames)

        def apply_fade_in(array, duration = 30):
            for i in range(duration):
                alpha = i / duration
                array[i] = cv2.addWeighted(array[i], alpha, np.zeros_like(array[i]), 1 - alpha, 0)
            return array


        def apply_fade_out(array, duration = 30):
            for i in range(duration):
                alpha = (duration - i) / duration
                array[num_frames - i - 1] = cv2.addWeighted(array[num_frames - i - 1], alpha,
                                                             np.zeros_like(array[num_frames - i - 1]), 1 - alpha, 0)
            return array

        frames = apply_fade_in(frames, fade_in_length)
        frames = apply_fade_out(frames, fade_out_length)

        return frames
