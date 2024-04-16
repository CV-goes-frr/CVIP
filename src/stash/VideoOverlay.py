import cv2
import numpy as np

def overlay_videos(video_path_1, video_path_2, output_path, resize_factor=4, x_offset=10, y_offset=10):
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    fps = min(fps1, fps2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width1, height1))

    for _ in range(total_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame2_resized = cv2.resize(frame2, (width2 // resize_factor, height2 // resize_factor))

        frame1[y_offset:y_offset + frame2_resized.shape[0],
        x_offset:x_offset + frame2_resized.shape[1]] = frame2_resized

        out.write(frame1)

    # If one video is longer, continue processing frames until both videos are exhausted
    while ret1 or ret2:
        if ret1:
            ret1, frame1 = cap1.read()
            out.write(frame1)
        if ret2:
            ret2, frame2 = cap2.read()
            frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)  # Black screen
            out.write(frame1)

    cap1.release()
    cap2.release()
    out.release()

    print("Overlay completed successfully!")

video_path_1 = "./media/jerry1.mp4"
video_path_2 = "./media/minion.mp4"
output_path = "output_video.mp4"

resize_factor = int(input("Enter the resize factor for the second video: "))
x_offset = int(input("Enter the x coordinate for the left upper corner of the second video: "))
y_offset = int(input("Enter the y coordinate for the left upper corner of the second video: "))

overlay_videos(video_path_1, video_path_2, output_path, resize_factor, x_offset, y_offset)