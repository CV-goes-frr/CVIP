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

        # Ensure fade lengths do not exceed the number of frames
        fade_in_length = min(self.fade_in_length, num_frames)
        fade_out_length = min(self.fade_out_length, num_frames)

        def fade_frame(i):
            if i < fade_in_length:  # Fade-in
                alpha = i / fade_in_length
                frames[i] = cv2.addWeighted(frames[i], alpha, np.zeros_like(frames[i]), 1 - alpha, 0)

            elif i >= num_frames - fade_out_length:  # Fade-out
                alpha = (fade_out_length - i) / fade_out_length
                frames[num_frames - i - 1] = cv2.addWeighted(frames[num_frames - i - 1], alpha,
                                                             np.zeros_like(frames[num_frames - i - 1]), 1 - alpha, 0)

            return frames[i]

        faded_frames = np.array([fade_frame(i) for i in range(num_frames)])

        return faded_frames