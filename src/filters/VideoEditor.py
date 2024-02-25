from typing import Dict

import cv2
import numpy as np
from multiprocessing import Pool

from .FaceDetection import FaceDetection
from .Filter import Filter


class VideoEditor(Filter):

    def __init__(self, filter_name: str):
        super().__init__()
        # self.class_map: Dict[str, type] = {
        #                                    "face_detection": FaceDetection}  # Список применяемых фильтров
        # Этот список, конечно, можно расширить
        # self.filter = self.class_map[filter_name]

    @staticmethod
    def apply(frames: np.ndarray, processes_limit: int, pool: Pool, filter: type, num_frames: int, width: int, height: int):
        print("Start VideoEditor")
        output = np.empty((num_frames, height, width, 3), np.uint8)  # Создаем список кадров
        index = 0
        for frame in frames:
            frame = filter.apply(frame, processes_limit, pool)  # Приемняем фильтр к каждому кадру
            output[index, :, :, :] = frame[0]  # cz we return [img] in filters
            index += 1
        return [output]


if __name__ == "__main__":
    processor = VideoEditor("face_detection")
    cap = cv2.VideoCapture("Patrick.mp4")  # Читаем изначальное видео

    # Находим характеристики изначального видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создаем cv2.VideoWriter, чтобы сделать выходное видео с характеристиками изначального
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    Pool = Pool(processes=2)

    # Делаем np.ndarray, который заполним кадрами видео
    frames = np.empty((num_frames, height, width, 3), np.uint8)
    index = 0

    while True:
        ret, frame = cap.read()  # читаем каждый кадр видео
        if not ret:
            break
        else:
            frames[index, :, :, :] = frame  # сохраняем каждый кадр видео
            index += 1

    new_frames = processor.apply(frames, 2, Pool, num_frames, width, height)  # Получаем кадры после применения фильтра

    for value in new_frames:
        out.write(value)  # Записываем видео по кадрам в cv2.VideoWriter

    out.release()  # Обязательно нужно закрыть
    cap.release()  # Иначе Иртегов пизды даст
