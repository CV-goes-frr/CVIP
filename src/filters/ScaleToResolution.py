from functools import lru_cache
from multiprocessing import Pool
from typing import List

import math
import numpy as np

from .Filter import Filter
from .decorators.bicubic_hermit_decorator import bicubic_hermit_cache


class ScaleToResolution(Filter):
    """
    A filter to scale an image to a specific resolution.
    """

    def __init__(self, size_x: str, size_y: str):
        """
        Initialize the ScaleToResolution filter with target sizes.

        Args:
            size_x (str): Target width.
            size_y (str): Target height.
        """
        super().__init__()
        self.size_x: int = int(size_x)
        self.size_y: int = int(size_y)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply the ScaleToResolution filter to the input image.

        Args:
            img (np.ndarray): Input image.
            processes_limit (int): Number of processes for parallelization.
            pool (Pool): Process pool.

        Returns:
            List[np.ndarray]: List containing the scaled image.
        """
        print("BICUBIC SCALE TO RESOLUTION IN PROCESS...")  # Print a message indicating the start of the process
        if self.cache:  # Check if there is a cached result
            print("USING CACHE...")  # Print a message indicating the use of cached result
            return self.cache  # Return the cached result

        input_height, input_width, _ = img.shape  # Get the dimensions of the input image
        width_scale_factor = self.size_x / input_width  # Calculate the scaling factor for the width
        height_scale_factor = self.size_y / input_height  # Calculate the scaling factor for the height
        new_width = self.size_x  # Set the new width after scaling
        new_height = self.size_y  # Set the new height after scaling
        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Initialize an empty upscaled image

        part_height = new_height // processes_limit  # Calculate the height of each processing part
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]  # Generate pixel coordinates
        parts = [coordinates[i:i + part_height] for i in range(0, len(coordinates), part_height)]  # Split coordinates into parts

        # Process pixels in parallel using multiprocessing pool
        processed_pixels = pool.starmap(self.process_pixel_resolution,
                                        [(x, y, width_scale_factor, height_scale_factor, input_width, input_height, img)
                                         for part in parts for (x, y) in part])

        # Assign processed pixels to the upscaled image
        for (x, y), pixel_value in zip(coordinates, processed_pixels):
            upscaled_image[y, x] = pixel_value

        if self.calls_counter > 1:  # Check if the method has been called more than once
            self.cache = [upscaled_image]  # Cache the upscaled image

        return [upscaled_image]  # Return the edited image as a list

    @staticmethod
    def process_pixel_resolution(x: int, y: int, scale_x: float, scale_y: float,
                                 input_width: int, input_height: int, img: np.ndarray) -> np.ndarray:
        """
        Process a single pixel during the scaling operation.

        Args:
            x (int): X position of the pixel.
            y (int): Y position of the pixel.
            scale_x (float): Scaling factor for the width.
            scale_y (float): Scaling factor for the height.
            input_width (int): Width of the input image.
            input_height (int): Height of the input image.
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed pixel value.
        """
        original_x = int(x / scale_x)  # Calculate the original x coordinate
        original_y = int(y / scale_y)  # Calculate the original y coordinate
        dx = original_x - math.floor(original_x)  # Calculate the fractional part of x
        dy = original_y - math.floor(original_y)  # Calculate the fractional part of y

        # Determine the coordinates of the four nearest pixels in the input image
        x_1 = min(max(math.floor(original_x) - 1, 0), input_width - 1)
        x_2 = min(max(math.floor(original_x), 0), input_width - 1)
        x_3 = min(max(math.floor(original_x) + 1, 0), input_width - 1)
        x_4 = min(max(math.floor(original_x) + 2, 0), input_width - 1)
        y_1 = min(max(math.floor(original_y) - 1, 0), input_height - 1)
        y_2 = min(max(math.floor(original_y), 0), input_height - 1)
        y_3 = min(max(math.floor(original_y) + 1, 0), input_height - 1)
        y_4 = min(max(math.floor(original_y) + 2, 0), input_height - 1)

        # Get the pixel values of the four nearest pixels
        pix11 = img[y_1, x_1]
        pix21 = img[y_1, x_2]
        pix31 = img[y_1, x_3]
        pix41 = img[y_1, x_4]
        pix12 = img[y_2, x_1]
        pix22 = img[y_2, x_2]
        pix32 = img[y_2, x_3]
        pix42 = img[y_2, x_4]
        pix13 = img[y_3, x_1]
        pix23 = img[y_3, x_2]
        pix33 = img[y_3, x_3]
        pix43 = img[y_3, x_4]
        pix14 = img[y_4, x_1]
        pix24 = img[y_4, x_2]
        pix34 = img[y_4, x_3]
        pix44 = img[y_4, x_4]

        # Perform bicubic interpolation using the four nearest pixels and the fractional parts of x and y
        arr1 = bicubic_hermit(pix11, pix21, pix31, pix41, dy)
        arr2 = bicubic_hermit(pix12, pix22, pix32, pix42, dy)
        arr3 = bicubic_hermit(pix13, pix23, pix33, pix43, dy)
        arr4 = bicubic_hermit(pix14, pix24, pix34, pix44, dy)

        # Perform bicubic interpolation along the x direction
        val = bicubic_hermit(arr1, arr2, arr3, arr4, dx)

        return val.astype(np.uint8)  # Return the processed pixel value

@bicubic_hermit_cache
def bicubic_hermit(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t):
    """
    Perform bicubic Hermite interpolation.

    Args:
        a (np.ndarray): Pixel value at position 'a'.
        b (np.ndarray): Pixel value at position 'b'.
        c (np.ndarray): Pixel value at position 'c'.
        d (np.ndarray): Pixel value at position 'd'.
        t: Interpolation factor.

    Returns:
        np.ndarray: Interpolated pixel value.
    """
    # Calculate the interpolated pixel value using the bicubic Hermite formula
    a_n = -1 * a / 2 + 3 * b / 2 - 3 * c / 2 + d / 2
    b_n = a - 5 * b / 2 + 2 * c - d / 2
    c_n = -1 * a / 2 + c / 2
    d_n = b

    return a_n * pow3(t) + b_n * t * t + c_n * t + d_n

@lru_cache(maxsize=128)
def pow3(t: int):
    """
    Calculate the cube of a number.

    Args:
        t (int): Input value.

    Returns:
        int: Cube of the input value.
    """
    return t * t * t