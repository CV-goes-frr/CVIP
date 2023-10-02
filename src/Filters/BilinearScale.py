import numpy as np
from typing import List

from .Filter import Filter


class BilinearScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("BILINEAR SCALE IN PROCESS...")
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)

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

                upscaled_image[y, x] = weight.astype(np.uint8)

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]
