from multiprocessing import Pool
from typing import List
import numpy as np

from .Filter import Filter

class NnScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.scale_factor: float = float(scale_factor)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image - List containing the edited image as a NumPy array
        """

        print("NN SCALE IN PROCESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        input_height, input_width, _ = img.shape  # Get the height and width of the input image
        new_width = int(input_width * self.scale_factor)  # Calculate the new width after scaling
        new_height = int(input_height * self.scale_factor)  # Calculate the new height after scaling
        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Create an empty upscaled image

        for y in range(new_height):  # Iterate over the rows of the upscaled image
            for x in range(new_width):  # Iterate over the columns of the upscaled image
                original_x = int(x / self.scale_factor)  # Calculate the original x coordinate
                original_y = int(y / self.scale_factor)  # Calculate the original y coordinate

                upscaled_image[y, x] = img[original_y, original_x]  # Copy the pixel from the original image

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = [upscaled_image]  # Cache the upscaled image

        return [upscaled_image]  # Return the upscaled image as a list
