import numpy as np
import math
from typing import List

from .Filter import Filter


class BicubicScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    def bicubic_hermit(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, t):
        a = -1*A/2 + 3*B/2 - 3*C/2 + D/2
        b = A - 5*B/2 + 2*C - D/2
        c = -1*A/2 + C/2
        d = B
        return a*t*t*t + b*t*t + c*t + d

    def apply(self, img: np.ndarray) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("BICUBIC SCALE IN PROCESS...")
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for w in range(new_width):
            for h in range(new_height):

                original_x = int(w/self.scale_factor)
                original_y = int(h/self.scale_factor)

                dx = original_x - math.floor(original_x)
                dy = original_y - math.floor(original_y)

                x1 = min(max(math.floor(original_x) - 1, 0), input_width - 1)
                x2 = min(max(math.floor(original_x), 0), input_width - 1)
                x3 = min(max(math.floor(original_x) + 1, 0), input_width - 1)
                x4 = min(max(math.floor(original_x) + 2, 0), input_width - 1)

                y1 = min(max(math.floor(original_y) - 1, 0), input_height - 1)
                y2 = min(max(math.floor(original_y), 0), input_height - 1)
                y3 = min(max(math.floor(original_y) + 1, 0), input_height - 1)
                y4 = min(max(math.floor(original_y) + 2, 0), input_height - 1)

                pix11 = img[y1, x1]
                pix21 = img[y1, x2]
                pix31 = img[y1, x3]
                pix41 = img[y1, x4]

                pix12 = img[y2, x1]
                pix22 = img[y2, x2]
                pix32 = img[y2, x3]
                pix42 = img[y2, x4]

                pix13 = img[y3, x1]
                pix23 = img[y3, x2]
                pix33 = img[y3, x3]
                pix43 = img[y3, x4]

                pix14 = img[y4, x1]
                pix24 = img[y4, x2]
                pix34 = img[y4, x3]
                pix44 = img[y4, x4]

                arr1 = self.bicubic_hermit(pix11, pix21, pix31, pix41, dy)
                arr2 = self.bicubic_hermit(pix12, pix22, pix32, pix42, dy)
                arr3 = self.bicubic_hermit(pix13, pix23, pix33, pix43, dy)
                arr4 = self.bicubic_hermit(pix14, pix24, pix34, pix44, dy)

                val = self.bicubic_hermit(arr1, arr2, arr3, arr4, dx)

                upscaled_image[h, w] = val.astype(np.uint8)

        if self.calls_counter > 1:
            self.cache = [upscaled_image]

        return [upscaled_image]
