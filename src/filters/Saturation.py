import cv2
import numpy as np
from multiprocessing import Pool
from typing import List

from .Filter import Filter


class Saturation(Filter):

    def __init__(self, saturation_factor):
        """
        Initializes the BilinearScale filter.

        Args:
            saturation_factor (str): Change saturate by saturation_factor as a string.

        Returns:
            None
        """

        super().__init__()  # Call the constructor of the parent class (Filter)
        self.log = "SATURATION FILTER IN PROGRESS..."
        self.saturation_factor: float = float(
            saturation_factor)  # Initialize the saturation_factor attribute with the given value

        self.change_saturate = lambda x: x * self.saturation_factor if x * self.saturation_factor <= 255 else 255
        # The function to change value of saturate

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Applies saturation filter to the input image.

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

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV of image

        hsv[..., 1] = np.array([[self.change_saturate(x) for x in row] for row in hsv[..., 1]])  # Change saturate of
        # image

        edited_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # From HSV to image

        return [edited_img]  # Return the edited image as a list
