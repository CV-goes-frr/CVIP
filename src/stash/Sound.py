from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip


def merge(video_path, audio_path):
    # Load audio
    audio = AudioFileClip(audio_path)

    # Load video
    video = VideoFileClip(video_path)

    # Merge audio and video
    video = video.set_audio(audio)

    # Save merged file
    video.write_videofile("merged.mp4")


def extract_audio(video_path):
    video = VideoFileClip(video_path)

    # Extract audio
    audio = video.audio

    # Save audio
    audio.write_audiofile("audio.mp3")
