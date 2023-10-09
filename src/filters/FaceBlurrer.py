import dlib
import cv2
import numpy as np
from imutils import face_utils
from typing import List

from .Filter import Filter

predictor_path = "src/Filters/shape_predictor_68_face_landmarks_GTX.dat"

class FaceBlurrer(Filter):

    def __init__(self, coef: str):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.coef = int(coef)

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:

        if self.cache:
            print("USING CACHE...")
            return self.cache

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            roi = img[y:y + h, x:x + w]

            blurred_face = self.apply_custom_gaussian_blur(roi, self.custom_gaussian_kernel(99, self.coef))

            if blurred_face is not None:
                img[y:y + h, x:x + w] = blurred_face


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