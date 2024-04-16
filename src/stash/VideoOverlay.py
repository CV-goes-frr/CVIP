import cv2


def overlay_videos(video_path_1, video_path_2, output_path):
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    fps = min(fps1, fps2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width1, height1))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame2_resized = cv2.resize(frame2, (width2 // 4, height2 // 4))

        x_offset = 10
        y_offset = height1 - frame2_resized.shape[0] - 10
        frame1[y_offset:y_offset + frame2_resized.shape[0],
        x_offset:x_offset + frame2_resized.shape[1]] = frame2_resized

        out.write(frame1)

    cap1.release()
    cap2.release()
    out.release()

    print("Overlay completed successfully!")


video_path_1 = "./media/jerry1.mp4"
video_path_2 = "./media/minion.mp4"
output_path = "output_video.mp4"

overlay_videos(video_path_1, video_path_2, output_path)
