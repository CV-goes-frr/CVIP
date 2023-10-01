import cv2
from typing import List, Dict

from .Filters.BilinearScale import BilinearScale
from .Filters.BicubicScale import BicubicScale
from .Filters.Crop import Crop
from .Filters.NnScale import NnScale
from .Filters.Merge import Merge
from .Filters.Duplicate import Duplicate


class Processor:

    def __init__(self):
        self.fin: str = ""
        self.class_map: Dict[str, type] = {"crop": Crop,
                                           "nn_scale": NnScale,
                                           "bilinear_scale": BilinearScale,
                                           "bicubic_scale": BicubicScale,
                                           "merge": Merge,
                                           "duplicate": Duplicate}

        # what in-labels should be already done for applying our filter with this out-label
        self.label_dependencies: Dict[str, List[str]] = {}

        # what filter is mapped for the label
        self.label_in_map: Dict[str, any] = {}

        self.inp_image: str = ""

    def process(self, label: str) -> List:
        # get all results from previous filters
        image: List = []
        if label != '-i':
            for prev_label in self.label_dependencies[label]:
                if prev_label not in self.label_dependencies and prev_label != '-i':
                    raise Exception("Label doesn't exist: " + prev_label)

                prev_result = self.process(prev_label)
                for img in prev_result:
                    image.append(img)
        else:
            return [cv2.imread(self.inp_image)]

        # now let our filter process all we've got from previous
        # print(len(image), "to", label)
        result: List = []
        if len(image) == 2:
            result = self.label_in_map[label].apply(image[0], image[1])
        elif len(image) == 1:
            result = self.label_in_map[label].apply(image[0])

        print(len(result), "result(s) from", label, "\n")
        to_return: List = []
        for img in result:
            to_return.append(img)

        return to_return
