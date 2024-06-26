import cv2
import numpy as np
import math
from multiprocessing import Pool
from settings import prefix
from typing import Any

from .Filter import Filter


class VideoOverlay(Filter):
    def __init__(
            self,
            path_to_vid2: str,
            resize_factor: int,
            x_offset: int,
            y_offset: int,
            longest: int
    ):
        """
            Initializes the VideoOverlay filter.

            Args:
                path_to_vid2 (str): Path to overlay's video.
                resize_factor (int): How many times decrease the overlay's video.
                x_offset (int): Offset overlay's video by x.
                y_offset (int): Offset overlay's video by y.
                longest (int): If longest == 1 then overlay by longest video else the other way around.

            Returns:
                None
        """
        super().__init__()
        self.log = "OVERLAYING VIDEOS IN PROCESS..."
        self.video_path2 = path_to_vid2
        self.resize_factor = int(resize_factor)
        self.x_offset = int(x_offset)
        self.y_offset = int(y_offset)
        self.longest = bool(int(longest))

    @staticmethod
    def increaseFps(
            input_frames: np.ndarray,
            in_fps: int,
            out_fps: int,
            num_frames: int
    ) -> list[Any]:
        """
            This method increases amount of frames to new fps.
            We need to increase the number of frames in the ratio of the output frames to the input frames.

            Args:
                input_frames (np.ndarray): Frames of the video where we change the FPS.
                in_fps (int): Initial fps.
                out_fps (int): Final fps.
                num_frames (int): Amount of frames of initial video

            Returns:
                List[np.ndarray]: List containing the edited amount of frames as a NumPy array.
        """

        fps = out_fps
        # If out_fps * 2 < in_fps then the ratio doesn't change just will be an extra integer
        if in_fps * 2 < out_fps:
            while in_fps * 2 < fps:
                fps = fps - in_fps

        gcd = math.gcd(fps, in_fps)

        # Find ratio
        denominator = fps // gcd
        rest = in_fps // gcd
        twice = denominator - rest

        if twice == rest:
            period = 1
        # When output fps = input fps * 2
        elif twice == 0:
            return list(input_frames)
        # So output fps is equal input fps
        else:
            period = int(round(rest / twice))
        # Once in period of frames we will multiply this frames

        output_frames = []
        # List of frames
        temp = period
        for i in range(num_frames):
            temp = temp - 1
            if temp == 0:
                temp = period
                for j in range(math.ceil(out_fps / in_fps)):
                    output_frames.append(input_frames[i])
                    # This frame needs to multiply one more than the others
            else:
                for j in range(math.ceil(out_fps / in_fps) - 1):
                    output_frames.append(input_frames[i])
                    # Multiply this frame if out_fps * 2 < in_fps * 2
        return output_frames

    @staticmethod
    def decreaseFps(
            input_frames: np.ndarray,
            in_fps: int,
            out_fps: int,
            num_frames: int
    ) -> list[Any]:
        """
            This method decreases amount of frames to new fps.
            We need to decrease the number of frames in the ratio of the output frames to the input frames.

            Args:
                input_frames (np.ndarray): Frames of the video where we change the FPS.
                in_fps (int): Initial fps.
                out_fps (int): Final fps.
                num_frames (int): Amount of frames of initial video

            Returns:
                List[np.ndarray]: List containing the edited amount of frames as a NumPy array.
        """

        gcd = math.gcd(out_fps, in_fps)

        # Find ratio
        denominator = in_fps // gcd

        if out_fps * 2 >= in_fps:
            # When output fps * 2 > input fps,
            # we need to delete fewer frames than rest
            rest = out_fps // gcd
            delete = denominator - rest

            period = int(round(rest / delete))
            # Once in period of frames we will delete a frame
            output_frames = []
            # List of frames

            temp = period + 1
            for i in range(num_frames):
                temp = temp - 1
                if temp == 0:
                    temp = period + 1
                    # This frame needs to delete
                else:
                    output_frames.append(input_frames[i])
                    # This frame needs to rest
        else:
            # When output fps * 2 > input fps,
            # we need to rest fewer frames than delete
            rest = out_fps // gcd
            delete = denominator - rest

            print(denominator, rest, delete)

            period = int(round(delete / rest))
            # Once in period of frames we will multiply this frames
            output_frames = []

            temp = period + 1
            for i in range(num_frames):
                temp = temp - 1
                if temp == 0:
                    temp = period + 1
                    output_frames.append(input_frames[i])
                    # This frame needs to rest

        return output_frames

    def apply(
            self,
            video1: np.ndarray,
            width1: int,
            height1: int,
            fps1: float,
            num_frames1: int,
            processes_limit: int,
            pool: Pool
    ) -> np.ndarray:
        """
            Applies VideoOverlay filter to the input video.

            Args:
                video1 (np.ndarray): Frames of the video where we use VideoOverlay.
                width1 (int): Width of video1.
                height1 (int): Height of video1.
                fps1 (float): FPS of video1.
                num_frames1 (int): Amount of frames of video1.
                processes_limit (int): Number of processes to use.
                pool (Pool): Pool of processes.

            Returns:
                List[np.ndarray]: List containing the edited frames as a NumPy array.
        """

        cap2 = cv2.VideoCapture(f'{prefix}/{self.video_path2}')
        # Information about video2
        width2, height2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps2 = int(round(cap2.get(cv2.CAP_PROP_FPS)))
        fps1 = int(round(fps1))
        total_frames2 = 0
        biggest_fps1 = fps1 >= fps2
        # If biggest_fps1 == True, then increaseFps, else decreaseFps

        video2 = []
        # List of frames
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            total_frames2 += 1
            video2.append(np.array(frame))
            # Read frames
        cap2.release()

        video2 = np.array(video2)

        duration1, duration2 = num_frames1 / fps1, total_frames2 / fps2

        longest_video_cap1 = duration1 >= duration2

        if biggest_fps1:
            video2 = self.increaseFps(video2, fps2, fps1, int(total_frames2))
        else:
            video2 = self.decreaseFps(video2, fps2, fps1, int(total_frames2))
        # Now fps of video1 is equal fps of video2
        video2 = np.array(video2)

        if self.longest:
            if longest_video_cap1:
                out = np.empty(
                    (int(num_frames1), height1, width1, 3),
                    np.uint8
                )
            else:
                out = np.empty(
                    (video2.shape[0], height1, width1, 3),
                    np.uint8
                )
        else:
            if longest_video_cap1:
                out = np.empty(
                    (video2.shape[0], height1, width1, 3),
                    np.uint8
                )
            else:
                out = np.empty(
                    (int(num_frames1), height1, width1, 3),
                    np.uint8
                )
        # Create output np.array

        frames_count = 0

        while True:
            if self.longest:
                if frames_count >= video1.shape[0] and frames_count >= video2.shape[0]:
                    break
                # When both videos are over, and we merge by the longest video
            else:
                if frames_count >= video1.shape[0] or frames_count >= video2.shape[0]:
                    break
                # When both videos are over, and we merge by the shortest video
            if video1.shape[0] <= frames_count < video2.shape[0]:
                # When video1 is over, but not video2
                frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
                frame2 = video2[frames_count]
                frame2_resized = cv2.resize(
                    frame2,
                    (width2 // self.resize_factor,
                     height2 // self.resize_factor)
                )
            elif video2.shape[0] <= frames_count < video1.shape[0]:
                # When video2 is over, but not video1
                frame1 = video1[frames_count]
                frame2_resized = np.zeros(
                    (height1 // self.resize_factor,
                     width1 // self.resize_factor, 3),
                    dtype=np.uint8
                )
            else:
                # When video1 and video2 aren't over
                frame1 = video1[frames_count]
                frame2 = video2[frames_count]
                frame2_resized = cv2.resize(
                    frame2,
                    (width2 // self.resize_factor,
                     height2 // self.resize_factor)
                )

            # Create output frame
            frame1[self.y_offset:self.y_offset + frame2_resized.shape[0],
            self.x_offset:self.x_offset + frame2_resized.shape[1]] = frame2_resized
            out[frames_count] = frame1
            frames_count += 1

        return out