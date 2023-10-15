import time
from multiprocessing import Pool
from typing import List, Dict

import cv2

from .filters.FaceDetection import FaceDetection
from .filters.BilinearScale import BilinearScale
from .filters.BicubicScale import BicubicScale
from .filters.Crop import Crop
from .filters.NnScale import NnScale
from .filters.Merge import Merge
from .filters.Duplicate import Duplicate
from .filters.FaceBlurrer import FaceBlurrer


class Processor:

    def __init__(self, processes_limit: int):
        self.fin_labels: List[str] = []
        if processes_limit > 4:
            processes_limit = 4
        self.processes_limit: int = processes_limit
        self.pool: Pool = Pool(processes=processes_limit)
        self.class_map: Dict[str, type] = {"crop": Crop,
                                           "nn_scale": NnScale,
                                           "bilinear_scale": BilinearScale,
                                           "bicubic_scale": BicubicScale,
                                           "merge": Merge,
                                           "duplicate": Duplicate,
                                           "face_blur": FaceBlurrer,
                                           "face_detection": FaceDetection}

        # what in-labels should be already done for applying our filter with this out-label
        self.label_dependencies: Dict[str, List[str]] = {}

        # what filter is mapped for the label
        self.label_in_map: Dict[str, any] = {}

        # what labels are going to be out-labels
        self.labels_to_out: Dict[str, List[str]] = {}

    def process(self, label: str) -> List:
        # get all results from previous filters
        image: List = []
        if label[0:3] != '-i=':
            for prev_label in self.label_dependencies[label]:
                prev_result = self.process(prev_label)
                for img in prev_result:
                    image.append(img)
        else:
            return [cv2.imread(f'{label[3::]}')]

        # now let our filter process all we've got from previous
        result: List = []
        start: float = time.time()
        # go deeper into dependencies
        if len(image) == 2:
            result = self.label_in_map[label].apply(image[0], image[1], self.processes_limit, self.pool)
        elif len(image) == 1:
            result = self.label_in_map[label].apply(image[0], self.processes_limit, self.pool)
        end: float = time.time()
        print("Time elapsed:", end - start)

        print(len(result), "result(s) from", label)
        ind = self.labels_to_out[label].index(label)
        print("Result with", ind, "index to return\n")
        to_return = [result[ind]]

        return to_return
