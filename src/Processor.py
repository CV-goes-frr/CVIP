import time
from multiprocessing import Pool
from typing import List, Dict

import cv2

from .filters.BilinearScale import BilinearScale
from .filters.BicubicScale import BicubicScale
from .filters.Crop import Crop
from .filters.NnScale import NnScale
from .filters.Merge import Merge
from .filters.Duplicate import Duplicate


class Processor:

    def __init__(self, processes_limit: int):
        self.fin_labels: List[str] = []
        if processes_limit > 8:
            processes_limit = 8
        self.processes_limit: int = processes_limit
        self.pool: Pool = Pool(processes=processes_limit)
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

        # what labels are going to be out-labels
        self.labels_to_out: Dict[str, List[str]] = {}

        self.inp_image: str = ""

    def process(self, label: str) -> List:
        # get all results from previous filters
        image: List = []
        if label != '-i':
            for prev_label in self.label_dependencies[label]:
                # if prev_label not in self.label_dependencies and prev_label != '-i':
                #     raise Exception("Label doesn't exist: " + prev_label)

                prev_result = self.process(prev_label)
                for img in prev_result:
                    image.append(img)
        else:
            return [cv2.imread(self.inp_image)]

        # now let our filter process all we've got from previous
        # print(len(image), "to", label)
        result: List = []
        start: float = time.time()
        if len(image) == 2:
            result = self.label_in_map[label].apply(image[0], image[1], self.processes_limit, self.pool)
        elif len(image) == 1:
            result = self.label_in_map[label].apply(image[0], self.processes_limit, self.pool)
        end: float = time.time()
        print("Time elapsed:", end - start)

        print(len(result), "result(s) from", label)
        to_return_indices = [out_label for out_label in self.labels_to_out[label] if out_label == label]

        if self.label_in_map[label].return_all:
            print("All images to return\n")
            return result

        ind = self.labels_to_out[label].index(label)
        print("Result with", ind, "index to return\n")
        to_return = [result[ind]]
        return to_return
