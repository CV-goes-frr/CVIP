from multiprocessing import Pool
from typing import List

import numpy as np

from src.filters.Filter import Filter


class NnScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool) -> List[np.ndarray]:
        """
        Apply signature for every Filter object. Method call edit input image and return new one.
        Shape of new img np.ndarray can be not the same as input shape.

        :param img: np.ndarray of pixels
        :param processes_limit: split the image into this number of pieces to process in parallel
        :param pool: processes pool
        :return: edited image
        """

        print("NN SCALE IN PROCESS...")
        if self.cache:
            print("USING CACHE...")
            return self.cache

        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)
        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)

                upscaled_image[y, x] = img[original_y, original_x]

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]
