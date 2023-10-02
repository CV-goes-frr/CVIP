import numpy as np
from typing import List
from multiprocessing import Pool

from .Filter import Filter


class BilinearScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    # Define process_pixel as a separate function outside the class
    @staticmethod
    def process_pixel(x, y, scale_factor, input_width, input_height, img):
        # This function calculates the pixel value for a given (x, y) coordinate
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)

        x1, y1 = int(original_x), int(original_y)
        x2, y2 = x1 + 1, y1 + 1

        x1 = min(max(x1, 0), input_width - 1)
        x2 = min(max(x2, 0), input_width - 1)
        y1 = min(max(y1, 0), input_height - 1)
        y2 = min(max(y2, 0), input_height - 1)

        alpha = original_x - x1
        beta = original_y - y1

        top_left = img[y1, x1]
        top_right = img[y1, x2]
        bottom_left = img[y2, x1]
        bottom_right = img[y2, x2]

        weight = (1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right + (
                1 - alpha) * beta * bottom_left + alpha * beta * bottom_right

        return weight.astype(np.uint8)

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        pool = Pool(processes=processes_limit)  # Create a Pool of 8 processes

        print("BILINEAR SCALE IN PROCESS...")
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Split the image into 8 equal parts
        part_height = new_height // processes_limit
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]
        parts = [coordinates[i:i+part_height] for i in range(0, len(coordinates), part_height)]

        # Use the Pool.map method to parallelize the pixel processing for each part
        processed_pixels = pool.starmap(self.process_pixel,
                                        [(x, y, self.scale_factor, input_width, input_height, img)
                                         for part in parts for (x, y) in part])

        # Convert the processed pixels back to an image
        for (x, y), pixel_value in zip(coordinates, processed_pixels):
            upscaled_image[y, x] = pixel_value

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]
