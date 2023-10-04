from functools import lru_cache
from typing import List
from multiprocessing import Pool

import numpy as np

from .Filter import Filter
from .decorators.bilinear_weight_decorator import bilinear_weight_cache


class BilinearScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    @staticmethod
    def process_pixel(x: int, y: int, scale_factor: float,
                      input_width: int, input_height: int, img: np.ndarray) -> np.ndarray:

        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)

        x_1, y_1 = int(original_x), int(original_y)
        x_2, y_2 = x_1 + 1, y_1 + 1

        x_1 = min(max(x_1, 0), input_width - 1)
        x_2 = min(max(x_2, 0), input_width - 1)
        y_1 = min(max(y_1, 0), input_height - 1)
        y_2 = min(max(y_2, 0), input_height - 1)

        alpha = original_x - x_1
        beta = original_y - y_1

        top_left = img[y_1, x_1]
        top_right = img[y_1, x_2]
        bottom_left = img[y_2, x_1]
        bottom_right = img[y_2, x_2]

        # weight = ((1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right
        #     + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right).astype(np.uint8)
        return weight_function(alpha, beta, top_left, top_right, bottom_left, bottom_right)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("BILINEAR SCALE IN PROCESS...")
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        part_height = new_height // processes_limit
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]
        parts = [coordinates[i:i+part_height] for i in range(0, len(coordinates), part_height)]

        processed_pixels = pool.starmap(self.process_pixel,
                                        [(x, y, self.scale_factor, input_width, input_height, img)
                                         for part in parts for (x, y) in part])

        for (x, y), pixel_value in zip(coordinates, processed_pixels):
            upscaled_image[y, x] = pixel_value

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]


@bilinear_weight_cache
def weight_function(alpha, beta, top_left: np.ndarray, top_right: np.ndarray,
                    bottom_left: np.ndarray, bottom_right: np.ndarray):
    return ((1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right
            + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right).astype(np.uint8)
