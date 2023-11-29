import time
from multiprocessing import Pool
from typing import List, Dict

import cv2

from src.filters.BilinearScale import BilinearScale
from src.filters.BicubicScale import BicubicScale
from src.filters.Crop import Crop
from src.filters.Duplicate import Duplicate
from src.filters.FaceBlurrer import FaceBlurrer
from src.filters.FaceDetection import FaceDetection
from src.filters.Merge import Merge
from src.filters.NnScale import NnScale
from src.filters.OverlayingMask import OverlayingMask
from settings import prefix


class Processor:

    def __init__(self, processes_limit: int):
        """
        :param processes_limit: number of processes in the pool
        """

        self.fin_labels: List[str] = []  # labels to create output files from
        if processes_limit > 4:  # if we create more than 4 processes, we can blow up machines without enough RAM
            processes_limit = 4
        self.processes_limit: int = processes_limit
        self.pool: Pool = Pool(processes=processes_limit)

        # dictionary to create filter objects
        self.class_map: Dict[str, type] = {"crop": Crop,
                                           "nn_scale": NnScale,
                                           "bilinear_scale": BilinearScale,
                                           "bicubic_scale": BicubicScale,
                                           "merge": Merge,
                                           "duplicate": Duplicate,
                                           "face_blur": FaceBlurrer,
                                           "face_detection": FaceDetection,
                                           "mask": OverlayingMask}

        # what in-labels should be already done for applying the filter with this out-label
        self.label_dependencies: Dict[str, List[str]] = {}

        # what filter is mapped for the label
        self.label_in_map: Dict[str, any] = {}

        # what labels are going to be out-labels
        self.labels_to_out: Dict[str, List[str]] = {}

    def process(self, label: str) -> List:
        """
        Applying a filter with out-label = label.

        :param label: the out-label of the filter
        :return: edited image(s)
        """

        # on every call we need to return only one image that is connected with our out-label
        dependency_ind = self.labels_to_out[label].index(label)  # so we get the index of our label

        # what label should we get from the dependencies to give the "label" result
        prev_label = self.label_dependencies[label][dependency_ind]

        if prev_label[0:3] != '-i=':
            prev_result = self.process(prev_label)  # process the previous label we need
        else:
            prev_result = [cv2.imread(f'{prefix}/{prev_label[3::]}')]  # or read the image

        # now let our filter process all we've got from previous
        result: List = []
        start: float = time.time()

        for prev_res in prev_result:
            res = self.label_in_map[label].apply(prev_res, self.processes_limit, self.pool)
            for r in res:
                result.append(r)

        end: float = time.time()
        print("Time elapsed:", end - start)

        return result
