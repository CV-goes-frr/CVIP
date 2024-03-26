from multiprocessing import Pool
from typing import List

import numpy as np

from .Filter import Filter

class NnScale(Filter):
    def __init__(self, scale_factor: str):
        """
        Initializes the NnScale filter.

        Args:
            scale_factor (str): Scale factor as a string.

        Returns:
            None
        """
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.scale_factor: float = float(scale_factor)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Applies the nearest neighbor scaling to the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.

        Returns:
            List[np.ndarray]: List containing the edited image as a NumPy array.
        """
        print("NN SCALE IN PROGRESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        # Get the height and width of the input image
        input_height, input_width, _ = img.shape

        # Calculate the new width and height after scaling
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        # Create an empty upscaled image
        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Iterate over the rows and columns of the upscaled image
        for y in range(new_height):
            for x in range(new_width):
                # Calculate the original x and y coordinates
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)

                # Copy the pixel from the original image to the upscaled image
                upscaled_image[y, x] = img[original_y, original_x]

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = [upscaled_image]  # Cache the upscaled image

        return [upscaled_image]  # Return the upscaled image as a list