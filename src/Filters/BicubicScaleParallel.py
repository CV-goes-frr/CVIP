import numpy as np
import math

from typing import List
from multiprocessing import Pool

from .Filter import Filter


class BicubicScale(Filter):

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor: float = float(scale_factor)

    @staticmethod
    def process_pixel(w: int, h: int, scale_factor: float,
                      input_width: int, input_height: int, img: np.ndarray):
        original_x = int(w / scale_factor)
        original_y = int(h / scale_factor)

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

        arr1 = bicubic_hermit(pix11, pix21, pix31, pix41, dy)
        arr2 = bicubic_hermit(pix12, pix22, pix32, pix42, dy)
        arr3 = bicubic_hermit(pix13, pix23, pix33, pix43, dy)
        arr4 = bicubic_hermit(pix14, pix24, pix34, pix44, dy)

        val = bicubic_hermit(arr1, arr2, arr3, arr4, dx)

        return val.astype(np.uint8)

    def apply(self, img: np.ndarray, processes_limit: int) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        pool = Pool(processes=processes_limit)  # Create a Pool of processes

        print("BICUBIC SCALE IN PROCESS...")
        input_height, input_width, _ = img.shape
        new_width = int(input_width * self.scale_factor)
        new_height = int(input_height * self.scale_factor)

        upscaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Split the image into processes_limit equal parts
        part_height = new_height // processes_limit
        coordinates = [(x, y) for x in range(new_width) for y in range(new_height)]
        parts = [coordinates[i:i + part_height] for i in range(0, len(coordinates), part_height)]

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


def bicubic_hermit(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t):
    a_n = -1*a/2 + 3*b/2 - 3*c/2 + d/2
    b_n = a - 5*b/2 + 2*c - d/2
    c_n = -1*a/2 + c/2
    d_n = b

    return a_n*t*t*t + b_n*t*t + c_n*t + d_n