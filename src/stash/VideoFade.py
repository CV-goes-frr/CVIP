import cv2
import numpy as np

def read_video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def apply_fade_in(frames, duration=30):
    num_frames = len(frames)
    fade_frames = min(duration, num_frames)
    for i in range(fade_frames):
        alpha = i / fade_frames
        frames[i] = cv2.addWeighted(frames[i], alpha, np.zeros_like(frames[i]), 1 - alpha, 0)
        print(frames[i])
    return frames

def apply_fade_out(frames, duration=30):
    num_frames = len(frames)
    fade_frames = min(duration, num_frames)
    for i in range(fade_frames):
        alpha = (fade_frames - i) / fade_frames
        frames[num_frames - i - 1] = cv2.addWeighted(frames[num_frames - i - 1], alpha, np.zeros_like(frames[num_frames - i - 1]), 1 - alpha, 0)
    return frames

def write_numpy_to_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def main(input_video_path, output_video_path, fade_in_duration=30, fade_out_duration=30):
    frames = read_video_to_numpy(input_video_path)
    frames = apply_fade_in(frames, fade_in_duration)
    frames = apply_fade_out(frames, fade_out_duration)
    write_numpy_to_video(frames, output_video_path)

if __name__ == "__main__":
    input_video_path = './media/jerry1.mp4'
    output_video_path = 'output_video.mp4'
    fade_in_duration = 30  # duration in frames
    fade_out_duration = 30  # duration in frames

    main(input_video_path, output_video_path, fade_in_duration, fade_out_duration)