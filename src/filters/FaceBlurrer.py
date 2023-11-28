from typing import List
from multiprocessing import Pool
import numpy as np
import dlib
from imutils import face_utils
import cv2

import sys, os
os.chdir(sys._MEIPASS)

from settings import prefix
from src.filters.Filter import Filter

PREDICTOR_PATH = "shape_predictor_81_face_landmarks.dat"


class FaceBlurrer(Filter):

    def __init__(self, coef: str):
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.detector = dlib.get_frontal_face_detector()  # Initialize the face detector
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Initialize the face landmarks predictor
        self.coef = int(coef)  # Convert the coefficient to an integer

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Face detection with dlib.get_frontal_face_detector().
        Creating a mask to outline the face silhouette based on specific facial landmarks.

        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit: we'll try to parallel it later
        :param pool: processes pool
        :return: edited image - List containing the edited image as a NumPy array
        """

        print("OUTLINING FACE SILHOUETTE...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        img_copy = np.copy(img)
        blurred_face = cv2.GaussianBlur(img_copy, (0, 0), self.coef)  # Apply Gaussian blur to the face

        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        rects = self.detector(gray, 0)  # Detect faces in the grayscale image

        for (i, rect) in enumerate(rects):  # Iterate over the detected faces
            shape = self.predictor(gray, rect)  # Get the facial landmarks for the current face
            shape = face_utils.shape_to_np(shape)  # Convert the landmarks to NumPy array

            # Define points to outline the face silhouette in the specified order
            face_silhouette = np.concatenate((
                shape[0:17],  # Points 0 to 16
                shape[78:79],  # Point 78
                shape[74:75],  # Point 74
                shape[79:80],  # Point 79
                shape[73:74],  # Point 73
                shape[72:73],  # Point 72
                shape[80:81],  # Point 80
                shape[71:72],  # Point 71
                shape[70:71],  # Point 70
                shape[69:70],  # Point 69
                shape[68:69],  # Point 68
                shape[76:77],  # Point 76
                shape[75:76],  # Point 75
                shape[77:78],  # Point 77
                shape[0:1]  # Point 0 (to close the loop)
            ), axis=0)

            mask = np.zeros_like(img_copy)  # Create a mask with the same shape as the image
            cv2.fillPoly(mask, [face_silhouette], (255, 255, 255))  # Fill the mask to outline face silhouette
            img_copy = np.where(mask != 0, blurred_face, img_copy)  # Set non-silhouette areas to black

        return [img_copy]  # Return the edited image as a list

    @staticmethod
    def reflect(self, p: np.array, x0: int, y0: int, x1: int, y1: int):
        """
        Point reflection relative to the line that is set by (x0, y0) and (x1, y1).

        :param self: self
        :param p: point to reflect (np.array(x, y)) - Point to be reflected
        :param x0: x of the first point of the line - X-coordinate of the first point of the line
        :param y0: y of the first point of the line - Y-coordinate of the first point of the line
        :param x1: x of the second point of the line - X-coordinate of the second point of the line
        :param y1: y of the second point of the line - Y-coordinate of the second point of the line
        :return: edited image - The reflected point as a NumPy array
        """

        dx = x1 - x0
        dy = y1 - y0

        a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
        b = 2 * dx * dy / (dx * dx + dy * dy)

        x2 = int(a * (p[0] - x0) + b * (p[1] - y0) + x0)
        y2 = int(b * (p[0] - x0) - a * (p[1] - y0) + y0)

        return np.array([x2, y2])
