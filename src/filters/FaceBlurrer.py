import math
from typing import List
from multiprocessing import Pool

import numpy as np
import dlib
from imutils import face_utils
import cv2

from .Filter import Filter

PREDICTOR_PATH = "src/filters/shape_predictor_68_face_landmarks_GTX.dat"


class FaceBlurrer(Filter):

    def __init__(self, coef: str):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.coef = int(coef)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Face detection with dlib.get_frontal_face_detector().
        Blurring faces according to jawline and reflected jawline (relative to the nose line).

        :param img: np.ndarray of pixels
        :param processes_limit: we'll try to parallel it later
        :param pool: processes pool
        :return: edited image
        """

        if self.cache:
            print("USING CACHE...")
            return self.cache

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            jawline = shape[0:17]

            x0 = shape[0][0]
            y0 = shape[0][1]
            x1 = shape[16][0]
            y1 = shape[16][1]
            for mirror_ind in range(1, 9):  # mirroring jawline
                jawline = np.append(jawline,
                                    np.array([np.array(self.reflect(self, jawline[17 - mirror_ind], x0, y0, x1, y1))]),
                                    axis=0)

            for mirror_ind in range(9, 0, -1):
                jawline = np.append(jawline,
                                    np.array([np.array(self.reflect(self, jawline[mirror_ind], x0, y0, x1, y1))]),
                                    axis=0)

            mask = np.zeros_like(img)
            cv2.fillPoly(mask, [jawline], (255, 255, 255))
            blurred_face = cv2.GaussianBlur(img, (0, 0), 30)
            img = np.where(mask != 0, blurred_face, img)

        return [img]

    @staticmethod
    def reflect(self, p: np.array, x0: int, y0: int, x1: int, y1: int):
        """
        Point reflection relative to the line that is set by (x0, y0) and (x1, y1).

        :param self: self
        :param p: point to reflect (np.array(x, y))
        :param x0: x of the first point of the line
        :param y0: y of the first point of the line
        :param x1: x of the second point of the line
        :param y1: y of the second point of the line
        :return: edited image
        """

        dx = x1 - x0
        dy = y1 - y0

        a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
        b = 2 * dx * dy / (dx * dx + dy * dy)

        x2 = int(a * (p[0] - x0) + b * (p[1] - y0) + x0)
        y2 = int(b * (p[0] - x0) - a * (p[1] - y0) + y0)

        return np.array([x2, y2])
