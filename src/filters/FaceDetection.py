import cv2
import numpy as np
from typing import List
from multiprocessing import Pool

from settings import BASE_DIR
from src.filters.Filter import Filter


HAARCASCADE_PATH = BASE_DIR + "/src/filters/haarcascade_frontalface_alt2.xml"


class FaceDetection(Filter):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class (Filter)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Face detection with cv2.CascadeClassifier().
        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit: we'll try to parallel it later
        :param pool: processes pool
        :return: List containing the edited image as a NumPy array
        """
        if self.cache: # Check if a cached result exists
            print("USING CACHE...")
            return self.cache # Return the cached result
        img_copy = np.copy(img)
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

        face_detect = cv2.CascadeClassifier(HAARCASCADE_PATH) # Detect faces in the grayscale image

        """
        Haar-feature - a Haar-like feature consists of dark regions and light regions. 
        It produces a single value by taking the difference of the sum of the intensities of the dark regions
        and the sum of the intensities of light regions
        """

        """
        Cascading is a particular case of ensemble learning based on the concatenation of several classifiers,
        using all information collected from the output from a given classifier
        as additional information for the next classifier in the cascade.
        """

        frontal_rect = face_detect.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)
        #detectMultiScale method returns boundary rectangles for the detected faces (i.e., x, y, w, h).


        for (x, y, w, h) in frontal_rect: # Rectangles are drawn around the detected faces
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return [img_copy] # Return the edited image as a list
