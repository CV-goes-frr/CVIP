import cv2
import numpy as np
import math

'''
self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # reading all the parameters
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
'''


def overlay_videos(video1, video_path_2, output_path, width1, height1, fps1, num_frames1, resize_factor=4, x_offset=10, y_offset=10,
                   longest=True):

    cap2 = cv2.VideoCapture(video_path_2)
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
    more_fps_1 = fps1 >= fps2
    longest_Video_cap1 = duration1 >= duration2
    longest_Video, shortest_Video = ((video1, fps1, width1, height1, num_frames1) if duration1 >= duration2 else (
    video2, fps2, width2, height2, total_frames2),
                                     (video2, fps2, width2, height2, total_frames2) if duration2 < duration1 else (
                                     video1, fps1, width1, height1, num_frames1))
    fps_ratio = math.ceil(shortest_Video[1] / longest_Video[1])

    if longest:
        out = np.empty((int(longest_Video[4] * fps_ratio), height1, width1, 3), np.uint8)
    else:
        out = np.empty((int(shortest_Video[4]), height1, width1, 3), np.uint8)
    frames_count = 0
    prev_ret, prev_frame = None, None

    while True:
        res = []

        if more_fps_1:
            res.append(video2[0])
            i = 0
            while i < video2.shape[0] - 1:
                frame1, frame2 = video2[i], video2[i+1]
                mean_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
                res.append(mean_frame)
                res.append(frame2)
                i += 1
        else:
            res.append(video1[0])
            i = 0
            while i < video1.shape[0] - 1:
                frame1, frame2 = video1[i], video1[i + 1]
                mean_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
                res.append(mean_frame)
                res.append(frame2)
                i += 1

    i = 0
    j = 0
    while True:
        if longest_Video_cap1:
            if frames_count % fps_ratio == 0:
                frame1 = video1[i]
                i += 1
                prev_frame = frame1
            else:
                frame1 = prev_frame
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

        if longest:
            if i >= video1.shape[0] and j >= video2.shape[0]:
                break
        else:
            if i >= video1.shape[0] or j >= video2.shape[0]:
                break

        if i >= video1.shape[0] and j < video2.shape[0]:
            frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
            if not longest_Video_cap1:
                frame2_resized = cv2.resize(frame2, (width2 // resize_factor, height2 // resize_factor))
        elif j >= video2.shape[0] and i < video1.shape[0]:
            frame2_resized = np.zeros((height1 // resize_factor, width1 // resize_factor, 3), dtype=np.uint8)
        else:
            frame2_resized = cv2.resize(frame2, (width2 // resize_factor, height2 // resize_factor))

        frame1[y_offset:y_offset + frame2_resized.shape[0],
        x_offset:x_offset + frame2_resized.shape[1]] = frame2_resized
        out[frames_count] = frame1
        frames_count += 1

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(video1.get(cv2.CAP_PROP_FPS)),
                            (out.shape[2], out.shape[1]))

    for i in range(out.shape[0]):
        data = out[i, :, :, :]
        video.write(data)
    video.release()

    print("Overlay completed successfully!")


video_path_2 = "jerry1.mp4"
video_path_1 = "minion.mp4"
output_path = "output_video.mp4"

cap1 = cv2.VideoCapture(video_path_1)
width1, height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = cap1.get(cv2.CAP_PROP_FPS)
total_frames1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)

video1 = []
while True:
    ret, frame = cap1.read()
    if not ret:
        break
    video1.append(np.array(frame))
cap1.release()
video2 = np.array(video1)

overlay_videos(video2, video_path_2, output_path, width1, height1, fps1, total_frames1, 4, 10, 10, True)
