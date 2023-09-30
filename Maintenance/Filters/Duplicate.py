import numpy as np
from typing import List


class Duplicate:

    def apply(self, img: np.ndarray) -> List[np.ndarray]:
        return [img, img]
