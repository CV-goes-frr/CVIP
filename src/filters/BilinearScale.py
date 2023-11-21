from typing import List
from multiprocessing import Pool
import numpy as np

from src.filters.Filter import Filter
from src.decorators.bilinear_weight_decorator import bilinear_weight_cache


class BilinearScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.scale_factor: float = float(scale_factor)  # Initialize the scale_factor attribute with the given value

    @staticmethod
    def process_pixel(x: int, y: int, scale_factor: float,
                      input_width: int, input_height: int, img: np.ndarray) -> np.ndarray:
        """
        Process a single pixel for bilinear scaling.

        :param x: x-coordinate of the pixel
        :param y: y-coordinate of the pixel
        :param scale_factor: how many times should we upscale the given image
        :param input_width: width of the input image
        :param input_height: height of the input image
        :param img: input image as a NumPy array
        :return: processed pixel as a NumPy array
        """

        original_x = int(x / scale_factor)  # Calculate the original x coordinate
        original_y = int(y / scale_factor)  # Calculate the original y coordinate

        x_1, y_1 = int(original_x), int(original_y)  # Determine the integer part of original coordinates
        x_2, y_2 = x_1 + 1, y_1 + 1  # Calculate the coordinates of the pixel to the right and the one below

        # Ensure that the coordinates are within the image boundaries
        x_1 = min(max(x_1, 0), input_width - 1)
        x_2 = min(max(x_2, 0), input_width - 1)
        y_1 = min(max(y_1, 0), input_height - 1)
        y_2 = min(max(y_2, 0), input_height - 1)

        alpha = original_x - x_1  # Calculate the alpha value for interpolation
        beta = original_y - y_1  # Calculate the beta value for interpolation

        top_left = img[y_1, x_1]  # Get the pixel value of the top-left corner
        top_right = img[y_1, x_2]  # Get the pixel value of the top-right corner
        bottom_left = img[y_2, x_1]  # Get the pixel value of the bottom-left corner
        bottom_right = img[y_2, x_2]  # Get the pixel value of the bottom-right corner

        return weight_function(alpha, beta, top_left, top_right, bottom_left,
                               bottom_right)  # Call the weight_function for interpolation

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """
        print("BILINEAR SCALE IN PROGRESS...")
        if self.cache:  # Check if a cached result exists
            print("USING CACHE...")
            return self.cache  # Return the cached result

        input_height, input_width, _ = img.shape  # Get the height and width of the input image
        new_width = int(input_width * self.scale_factor)  # Calculate the new width after scaling
        new_height = int(input_height * self.scale_factor)  # Calculate the new height after scaling

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Create an empty upscaled image

        part_height = new_height // processes_limit  # Calculate the height of each processing part
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]  # Generate pixel coordinates
        parts = [coordinates[i:i + part_height] for i in
                 range(0, len(coordinates), part_height)]  # Split coordinates into parts

        processed_pixels = pool.starmap(self.process_pixel,
                                        [(x, y, self.scale_factor, input_width, input_height, img)
                                         for part in parts for (x, y) in part])  # Process pixels in parallel

        for (x, y), pixel_value in zip(coordinates, processed_pixels):  # Assign processed pixels to the upscaled image
            upscaled_image[y, x] = pixel_value

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = [upscaled_image]  # Cache the upscaled image

        return [upscaled_image]  # Return the edited image as a list


@bilinear_weight_cache  # Apply the bilinear_weight_cache decorator to the following function
def weight_function(alpha, beta, top_left: np.ndarray, top_right: np.ndarray,
                    bottom_left: np.ndarray, bottom_right: np.ndarray):
    """
    Calculate the weighted sum of pixel values for bilinear scaling.

    :param alpha: Alpha value (interpolation factor for x-direction, ranging from 0 to 1)
    :param beta: Beta value (interpolation factor for y-direction, ranging from 0 to 1)
    :param top_left: Top-left pixel value as a NumPy array
    :param top_right: Top-right pixel value as a NumPy array
    :param bottom_left: Bottom-left pixel value as a NumPy array
    :param bottom_right: Bottom-right pixel value as a NumPy array
    :return: Weighted pixel value as a NumPy array

    This function computes the weighted sum of the four corner pixels using the given interpolation factors.

    - 'alpha' and 'beta' represent the fractional part of the x and y coordinates.
    - 'top_left', 'top_right', 'bottom_left', and 'bottom_right' are the pixel values at the four corners.

    The interpolation formula combines these corner values based on 'alpha' and 'beta':
    - (1 - alpha) * (1 - beta) * top_left: Contribution from the top-left pixel
    - alpha * (1 - beta) * top_right: Contribution from the top-right pixel
    - (1 - alpha) * beta * bottom_left: Contribution from the bottom-left pixel
    - alpha * beta * bottom_right: Contribution from the bottom-right pixel

    The final result is the weighted sum of these contributions, representing the pixel value at the given (alpha, beta).
    The result is cast to uint8 data type, representing an 8-bit grayscale or color pixel.
    """

    return ((1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right
            + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right).astype(np.uint8)
