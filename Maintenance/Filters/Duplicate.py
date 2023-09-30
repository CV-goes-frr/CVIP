import sys
import os
import errno

import numpy as np


class Duplicate():

    def apply(self, img):
        return [img, img]
