from multiprocessing import Pool

import cv2
import numpy as np

from src.stash.panorama.PanoramicMerge import PanoramicMerge


def main():
    cap = cv2.VideoCapture("videoTest.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # reading all the parameters
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # create prev_result
    prev_result = [np.empty((num_frames, height, width, 3), np.uint8)]
    index = 0

    while True:
        ret, frame = cap.read()  # reading frame by frame
        if not ret:
            break
        else:
            prev_result[0][index, :, :, :] = frame
            index += 1
    cap.release()

    cv2.imwrite("fromVideo.png", apply(prev_result[0], 2, None))


def apply(frames: np.ndarray, processes_limit: int, pool: Pool) -> np.ndarray:
    # print(frames.shape)
    step = 30
    result = frames[0]

    for frame_index in range(step, len(frames), step):
        result = PanoramicMerge.process(result, frames[frame_index])

    return result


if __name__ == "__main__":
    main()
