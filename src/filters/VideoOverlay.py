from multiprocessing import Pool

import numpy as np
import cv2
import math

from settings import prefix
from .Filter import Filter

class VideoOverlay(Filter):
    def __init__(self, path_to_vid2, resize_factor, x_offset, y_offset, longest):
        super().__init__()
        self.log = "OVERLAYING VIDEOS IN PROCESS..."
        self.video_path2 = path_to_vid2
        self.resize_factor = int(resize_factor)
        self.x_offset = int(x_offset)
        self.y_offset = int(y_offset)
        self.longest = bool(int(longest))

    def apply(self, video1: np.ndarray, width1, height1, fps1, num_frames1, processes_limit: int, pool: Pool) -> np.ndarray:
        cap2 = cv2.VideoCapture(f'{prefix}/{self.video_path2}')
        width2, height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        total_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
        video2 = []
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            video2.append(np.array(frame))
        cap2.release()
        video2 = np.array(video2)

        duration1, duration2 = num_frames1 / fps1, total_frames2 / fps2
        longest_Video, shortest_Video = ((video1, fps1, width1, height1, num_frames1) if duration1 >= duration2 else (
            video2, fps2, width2, height2, total_frames2),
                                         (video2, fps2, width2, height2, total_frames2) if duration2 < duration1 else (
                                             video1, fps1, width1, height1, num_frames1))
        longest_Video_cap1 = duration1 >= duration2
        fps_ratio = math.ceil(shortest_Video[1] / longest_Video[1])

        if self.longest:
            if longest_Video_cap1:
                out = np.empty((int(longest_Video[4]), height1, width1, 3), np.uint8)
            else:
                out = np.empty((int(longest_Video[4] * fps_ratio), height1, width1, 3), np.uint8)
        else:
            if longest_Video_cap1:
                out = np.empty((int(shortest_Video[4]), height1, width1, 3), np.uint8)
            else:
                out = np.empty((int(shortest_Video[4]), height1, width1, 3), np.uint8)
        frames_count = 0
        prev_ret, prev_frame = None, None
        i = 0
        j = 0
        while True:
            if longest_Video_cap1:
                frame1 = video1[i]
                i += 1
                if j < video2.shape[0]:
                    frame2 = video2[j]
                    j += 1
                else:
                    frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)
            else:
                if frames_count % fps_ratio == 0:
                    frame2 = video2[j]
                    j += 1
                    prev_frame = frame2
                else:
                    frame2 = prev_frame
                if i < video1.shape[0]:
                    frame1 = video1[i]
                    i += 1
                else:
                    frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)

            if self.longest:
                if i >= video1.shape[0] and j >= video2.shape[0]:
                    break
            else:
                if i >= video1.shape[0] or j >= video2.shape[0]:
                    break

            if i >= video1.shape[0] and j < video2.shape[0]:
                frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
                if not longest_Video_cap1:
                    frame2_resized = cv2.resize(frame2, (width2 // self.resize_factor, height2 // self.resize_factor))
            elif j >= video2.shape[0] and i < video1.shape[0]:
                frame2_resized = np.zeros((height1 // self.resize_factor, width1 // self.resize_factor, 3), dtype=np.uint8)
            else:
                frame2_resized = cv2.resize(frame2, (width2 // self.resize_factor, height2 // self.resize_factor))

            frame1[self.y_offset:self.y_offset + frame2_resized.shape[0],
            self.x_offset:self.x_offset + frame2_resized.shape[1]] = frame2_resized
            out[frames_count] = frame1
            frames_count += 1

        return out
