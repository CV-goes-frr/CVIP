import cv2
import numpy as np
from typing import List
from multiprocessing import Pool

from .Filter import Filter


HAARCASCADE_PATH = "src/filters/haarcascade_frontalface_alt2.xml"


class FaceDetection(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detect = cv2.CascadeClassifier(HAARCASCADE_PATH)

        frontal_rect = face_detect.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in frontal_rect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return [img]
