import sys
import os
import errno

import numpy as np


class Crop():

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
    
    def apply(self, img):
        input_height, input_width, _ = img.shape
        if (self.x1 > input_width or self.y1 > input_height or self.x2 > input_width or self.y2 > input_height or
            (self.x1 >= self.x2) or (self.y1 >= self.y2) or self.x1 < 0 or self.x2 < 0 or self.y1 < 0 or self.y2 < 0 or
            type(self.x1) != int or type(self.x2) != int or type(self.y1) != int or type(self.y2) != int):
            raise Exception("Wrong crop parameters: " + self.x1 + ' ' + self.y1 + ' ' + self.x2 + ' ' + self.y2)
            
        cropped_image = img[self.y1:self.y2, self.x1:self.x2]
        return [cropped_image]
