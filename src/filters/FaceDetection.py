from multiprocessing import Pool
from typing import List

import cv2
import dlib
import numpy as np

from .Filter import Filter


class FaceDetection(Filter):
    def __init__(self):
        """
        Initializes the FaceDetection filter.

        Args:
            None

        Returns:
            None
        """
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.log = "FACE DETECTION IN PROGRESS..."

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Applies face detection using the dlib frontal face detector.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.

        Returns:
            List[np.ndarray]: List containing the edited image as a NumPy array.
        """

        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        # Create a copy of the input image
        img_copy = np.copy(img)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # Initialize the face detector
        detector = dlib.get_frontal_face_detector()

        # Detect faces in the grayscale image
        rects = detector(gray, 0)

        # Draw rectangles around the detected faces
        for (_, rect) in enumerate(rects):
            cv2.rectangle(img_copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        return [img_copy]  # Return the edited image as a list