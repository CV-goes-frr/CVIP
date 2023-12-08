import cv2
import dlib
import numpy as np
from typing import List
from multiprocessing import Pool

from .Filter import Filter


class FaceDetection(Filter):
    def __init__(self):
        super().__init__()  # Call the constructor of the parent class (Filter)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Face detection with dlib frontal face detector.
        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit: we'll try to parallel it later
        :param pool: processes pool
        :return: List containing the edited image as a NumPy array
        """

        print("FACE DETECTION IN PROGRESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        img_copy = np.copy(img)
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

        detector = dlib.get_frontal_face_detector()  # Initialize the face detector
        rects = detector(gray, 0)

        for (_, rect) in enumerate(rects):  # Iterate over the detected faces
            cv2.rectangle(img_copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        return [img_copy]  # Return the edited image as a list
