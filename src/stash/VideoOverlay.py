import cv2
import numpy as np
import math


def overlay_videos(video_path_1, video_path_2, output_path, resize_factor=4, x_offset=10, y_offset=10, longest=True):
    cap1, cap2 = cv2.VideoCapture(video_path_1), cv2.VideoCapture(video_path_2)
    width1, height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    total_frames1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)

    width2, height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)

    duration1, duration2 = total_frames1 / fps1, total_frames2 / fps2
    longest_Video, shortest_Video = cap1 if duration1 >= duration2 else cap2, cap2 if duration2 < duration1 else cap1
    longest_Video_cap1 = duration1 >= duration2
    fps_ratio = math.ceil(shortest_Video.get(cv2.CAP_PROP_FPS) / longest_Video.get(cv2.CAP_PROP_FPS))

    if longest:
        out = np.empty((int(longest_Video.get(cv2.CAP_PROP_FRAME_COUNT) * fps_ratio), height1, width1, 3), np.uint8)
    else:
        out = np.empty((int(shortest_Video.get(cv2.CAP_PROP_FRAME_COUNT)), height1, width1, 3), np.uint8)
    frames_count = 0
    prev_ret, prev_frame = None, None
    while True:
        if longest_Video_cap1:
            if frames_count % fps_ratio == 0:
                ret1, frame1 = cap1.read()
                prev_ret, prev_frame = ret1, frame1
            else:
                ret1, frame1 = prev_ret, prev_frame
            ret2, frame2 = cap2.read()
        else:
            if frames_count % fps_ratio == 0:
                ret2, frame2 = cap2.read()
                prev_ret, prev_frame = ret2, frame2
            else:
                ret2, frame2 = prev_ret, prev_frame
            ret1, frame1 = cap1.read()

        if longest:
            if not ret1 and not ret2:
                break
        else:
            if not ret1 or not ret2:
                break

        if not ret1 and ret2:
            frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
            if not longest_Video_cap1:
                frame2_resized = cv2.resize(frame2, (width2 // resize_factor, height2 // resize_factor))
        elif not ret2 and ret1:
            frame2_resized = np.zeros((height1 // resize_factor, width1 // resize_factor, 3), dtype=np.uint8)
        else:
            frame2_resized = cv2.resize(frame2, (width2 // resize_factor, height2 // resize_factor))

        frame1[y_offset:y_offset + frame2_resized.shape[0],
        x_offset:x_offset + frame2_resized.shape[1]] = frame2_resized
        out[frames_count] = frame1
        frames_count += 1

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(shortest_Video.get(cv2.CAP_PROP_FPS)),
                            (out.shape[2], out.shape[1]))
    for i in range(out.shape[0]):
        data = out[i, :, :, :]
        video.write(data)
    video.release()

    print("Overlay completed successfully!")


video_path_1 = "./media/jerry1.mp4"
video_path_2 = "./media/minion.mp4"
output_path = "output_video.mp4"

overlay_videos(video_path_1, video_path_2, output_path, 4, 10, 10, False)
