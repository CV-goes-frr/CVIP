import numpy as np
from typing import List
# from functools import lru_cache

from .Filter import Filter
# from .NpArrayDecorator import npArrToTuple


class Crop(Filter):

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        super().__init__()
        self.x1: int = int(x1)
        self.y1: int = int(y1)
        self.x2: int = int(x2)
        self.y2: int = int(y2)

    # def __hash__(self):
    #     return hash(tuple([x.tostring() for x in self.cache]))

    # @npArrToTuple
    #@lru_cache()
    def apply(self, img: np.ndarray) -> List[np.ndarray]:
        if self.cache:
            print("USING CACHE...")
            return self.cache

        print("CROP IN PROCESS...")
        input_height, input_width, _ = img.shape
        if (self.x1 > input_width or self.y1 > input_height or self.x2 > input_width or self.y2 > input_height) or (
                self.x1 >= self.x2 or self.y1 >= self.y2) or (
                self.x1 < 0 or self.x2 < 0 or self.y1 < 0 or self.y2 < 0) or (
                type(self.x1) != int or type(self.x2) != int or type(self.y1) != int or type(self.y2) != int):
            raise Exception(
                "Wrong crop parameters: " + str(self.x1) + ' ' + str(self.y1) + ' ' + str(self.x2) + ' ' + str(self.y2))

        result = [img[self.y1:self.y2, self.x1:self.x2]]

        if self.calls_counter > 1:
            self.cache = result

        return result
