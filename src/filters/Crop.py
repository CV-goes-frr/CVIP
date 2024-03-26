from multiprocessing import Pool
from typing import List

import numpy as np

from .Filter import Filter

class Crop(Filter):

    def __init__(self, x_1: str, y_1: str, x_2: str, y_2: str):
        """
        Initializes the Crop filter.

        Args:
            x_1 (str): X-coordinate of the upper-left corner of the cropped area.
            y_1 (str): Y-coordinate of the upper-left corner of the cropped area.
            x_2 (str): X-coordinate of the lower-right corner of the cropped area.
            y_2 (str): Y-coordinate of the lower-right corner of the cropped area.

        Returns:
            None
        """
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.x_1: int = int(x_1)  # Convert x_1 to an integer
        self.y_1: int = int(y_1)  # Convert y_1 to an integer
        self.x_2: int = int(x_2)  # Convert x_2 to an integer
        self.y_2: int = int(y_2)  # Convert y_2 to an integer

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Applies cropping to the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.

        Returns:
            List[np.ndarray]: List containing the cropped image as a NumPy array.
        """
        print("CROP IN PROGRESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        input_height, input_width, _ = img.shape  # Get the height and width of the input image

        # Check if the crop parameters are valid
        if (self.x_1 > input_width or self.y_1 > input_height or self.x_2 > input_width or self.y_2 > input_height) or (
                self.x_1 >= self.x_2 or self.y_1 >= self.y_2) or (
                self.x_1 < 0 or self.x_2 < 0 or self.y_1 < 0 or self.y_2 < 0) or (
                type(self.x_1) != int or type(self.x_2) != int or type(self.y_1) != int or type(self.y_2) != int):
            raise Exception(
                "Wrong crop parameters: " + str(self.x_1) + ' ' + str(self.y_1) + ' ' + str(self.x_2) + ' ' + str(self.y_2))

        # Perform cropping on the input image
        result = [img[self.y_1:self.y_2, self.x_1:self.x_2]]

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = result  # Cache the cropped image

        return result  # Return the cropped image as a list