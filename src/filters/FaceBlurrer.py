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

        if self.cache:
            print("USING CACHE...")
            return self.cache

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            jawline_points = shape[0:17]
            mask = np.zeros_like(img)
            cv2.fillConvexPoly(mask, jawline_points, (255, 255, 255))
            blurred_face = cv2.GaussianBlur(img, (0, 0), 30)
            img = np.where(mask != 0, blurred_face, img)

        return [img]

    @staticmethod
    def custom_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                         np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_custom_gaussian_blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        height, width, channels = img.shape
        ksize = kernel.shape[0]
        pad = ksize // 2

        padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        blurred_img = np.zeros_like(img, dtype=float)

        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                for c in range(channels):
                    region = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, c]
                    blurred_img[i - pad, j - pad, c] = np.sum(region * kernel)

        return blurred_img.astype(np.uint8)
