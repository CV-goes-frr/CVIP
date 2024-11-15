import cv2
import numpy as np
import time
from multiprocessing import Pool
from moviepy.editor import VideoFileClip
from typing import Dict, List

from settings import prefix
from .filters.BilinearScale import BilinearScale
from .filters.Crop import Crop
from .filters.FaceBlurrer import FaceBlurrer
from .filters.FaceDetection import FaceDetection
from .filters.FeatureMatching import FeatureMatching
from .filters.MotionTracking import MotionTracking
from .filters.NnScale import NnScale
from .filters.OverlayingMask import OverlayingMask
from .filters.Saturation import Saturation
from .filters.ScaleToResolution import ScaleToResolution
from .filters.VideoEditor import VideoEditor
from .filters.Flip import Flip
from .filters.VideoOverlay import VideoOverlay
from src.exceptions.WrongParameters import WrongParametersException
from src.filters.VideoToPanorama import VideoToPanorama
from .filters.VideoReverse import VideoReverse
from .filters.FadeEffect import FadeEffect


class Processor:

    def __init__(self, processes_limit: int, video_editing: bool):
        """
        Initializes Processor object with processes_limit and video_editing flag.

        Args:
            processes_limit (int): Number of processes in the pool.
            video_editing (bool): Flag indicating video editing mode.
        """
        self.video_editing = video_editing  # editing a video will change process
        self.num_frames = 0  # we need to store them when we open the video
        # self.width = 0
        # self.height = 0
        self.fps = 0
        self.audio = None

        self.fin_labels: List[str] = []  # labels to create output files from
        if processes_limit > 4:  # if we create more than 4 processes, we can blow up machines without enough RAM
            processes_limit = 4
        self.processes_limit: int = processes_limit
        self.pool: Pool = Pool(processes=processes_limit)

        # dictionary to create filter objects
        self.class_map: Dict[str, type] = {"crop": Crop,
                                           "nn_scale_with_factor": NnScale,
                                           "bilinear_scale_with_factor": BilinearScale,
                                           "scale_to_resolution": ScaleToResolution,
                                           "face_blur": FaceBlurrer,
                                           "face_detection": FaceDetection,
                                           "mask": OverlayingMask,
                                           "motion_tracking": MotionTracking,
                                           "feature_matching": FeatureMatching,
                                           "panorama": VideoToPanorama,
                                           "video_overlay": VideoOverlay,
                                           "reverse": VideoReverse,
                                           "flip": Flip,
                                           "fade": FadeEffect,
                                           "saturation": Saturation}

        # what in-labels should be already done for applying the filter with this out-label
        self.label_dependencies: Dict[str, List[str]] = {}

        # what filter is mapped for the label
        self.label_in_map: Dict[str, any] = {}

        # what labels are going to be out-labels
        self.labels_to_out: Dict[str, List[str]] = {}

    def process(self, label: str) -> List:
        """
        Applying a filter with out-label = label.

        Args:
            label (str): The out-label of the filter.

        Returns:
            List: Edited image(s).
        """
        # on every call we need to return only one image that is connected with our out-label
        dependency_ind = self.labels_to_out[label].index(label)  # so we get the index of our label

        # what label should we get from the dependencies to give the "label" result
        prev_label = self.label_dependencies[label][dependency_ind]

        if prev_label[0:3] != '-i=':
            prev_result = self.process(prev_label)  # process the previous label we need
        else:
            if self.video_editing:
                # read the video
                cap = cv2.VideoCapture(f'{prefix}/{prev_label[3::]}')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # reading all the parameters
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.audio = VideoFileClip(f'{prefix}/{prev_label[3::]}').audio
                # print(self.audio)

                # create prev_result
                prev_result = [np.empty((self.num_frames, height, width, 3), np.uint8)]
                index = 0

                while True:
                    ret, frame = cap.read()  # reading frame by frame
                    if not ret:
                        break
                    else:
                        prev_result[0][index, :, :, :] = frame
                        index += 1
                cap.release()
                # self.num_frames = len(prev_result[0])
            else:
                prev_result = [cv2.imread(f'{prefix}/{prev_label[3::]}')]  # or read the image

        # now let our filter process all we've got from previous
        result: List = []
        start: float = time.time()

        for prev_res in prev_result:
            self.label_in_map[label].start_log()
            if self.video_editing:  # applying filter frame by frame with VideoEditor class
                try:
                    height, width, _ = prev_res[0].shape  # get the shape of the first frame
                    res = VideoEditor.apply(prev_res, self.processes_limit, self.pool, self.label_in_map[label],
                                            self.num_frames, width, height, self.fps)
                except ValueError as e:
                    raise WrongParametersException(str(type(self.label_in_map[label])), str(e))
            else:  # apply operation for the image
                res = self.label_in_map[label].apply(prev_res, self.processes_limit, self.pool)

            for r in res:
                result.append(r)

        end: float = time.time()
        print("Time elapsed:", end - start)
        print()

        return result
