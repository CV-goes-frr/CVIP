import numpy as np
from typing import List

from .Filter import Filter


class NnScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("NN SCALE IN PROCESS...")
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
