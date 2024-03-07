import cv2
import numpy as np
from multiprocessing import Pool
import dlib
from imutils import face_utils

from .Filter import Filter
from settings import prefix
from .linal.RtcUmeyama import RtcUmeyama
from ..exceptions.NoFace import NoFaceException

PREDICTOR_PATH = "shape_predictor_81_face_landmarks.dat"


class OverlayingMask(Filter):
    def __init__(self, mask_name: str):
        super().__init__()  # Call the constructor of the parent class (Filter)
        self.log = "OVERLAYING MASKING IN PROGRESS..."

        self.detector = dlib.get_frontal_face_detector()  # Initialize the face detector
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Initialize the face landmarks predictor
        self.coef = 1.25

        # ## Find points on the mask
        self.mask_image = cv2.imread(f'{prefix}/{mask_name}')
        # Find landmarks on the mask
        mask_gray = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        rects = self.detector(mask_gray, 0)  # Detect faces in the grayscale image

        if len(rects) == 0:
            raise NoFaceException(mask_name)  # if there is no face on mask_image

        for (i, rect) in enumerate(rects):
            mask_shape = self.predictor(mask_gray, rect)  # Get the facial landmarks for the current face
        self.mask_landmarks = face_utils.shape_to_np(mask_shape)  # Convert the landmarks to NumPy array

        self.jaw_mark_l = 1
        self.jaw_mark_r = 15

        self.mask_detection_confidence = 0.5
        self.mask_tracking_confidence = 0.5
        self.mask_max_faces = 1

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        """
        :param self: self
        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit:we'll try to parallel it later
        :param pool: processes pool
        :return: edited image - List containing the edited image as a NumPy array
        """

        if self.cache:
            print("USING CACHE...")
            return self.cache

        # Shape of the mask we need to calculate transformation
        h_mask, w_mask, c_mask = img.shape

        # ## Find points on target faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        rects = self.detector(gray, 0)  # Detect faces in the grayscale image

        # Process all faces
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)  # Get the facial landmarks for the current face
            shape = face_utils.shape_to_np(shape)  # Convert the landmarks to NumPy array

            R, t, c = RtcUmeyama(shape, self.mask_landmarks)  # Calculating rotation, translation and scale

            mask_copy = self.mask_image.copy()

            # Form am Affine transformation matrix
            A = np.concatenate((c * R, np.expand_dims(t, axis=1)), axis=1)
            mask_copy = cv2.warpAffine(mask_copy, A, (w_mask, h_mask))  # Edit mask image

            face_silhouette = np.concatenate((
                shape[0:17],   # Points 0 to 16
                shape[78:79],  # Point 78
                shape[74:75],  # Point 74
                shape[79:80],  # Point 79
                shape[73:74],  # Point 73
                shape[72:73],  # Point 72
                shape[80:81],  # Point 80
                shape[71:72],  # Point 71
                shape[70:71],  # Point 70
                shape[69:70],  # Point 69
                shape[68:69],  # Point 68
                shape[76:77],  # Point 76
                shape[75:76],  # Point 75
                shape[77:78],  # Point 77
                shape[0:1]     # Point 0 (to close the loop)
            ), axis=0)

            # Create a poly on the face, where we'll change pixels to mask's pixels
            mask_poly = np.zeros_like(mask_copy)  # Create a mask with the same shape as the image
            cv2.fillPoly(mask_poly, [face_silhouette], (255, 255, 255))  # Fill the mask to outline face silhouette
            img = np.where(mask_poly != 0, mask_copy, img)  # Set non-silhouette areas to black

        if self.calls_counter > 1:
            self.cache = [img]

        return [img]  # Return the edited image as a list

    @staticmethod
    def scale(im: np.ndarray, nR: np.array, nC: np.array):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns
        return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]

    @staticmethod
    def reflect(self, p: np.array, x0: int, y0: int, x1: int, y1: int):
        """
        Point reflection relative to the line that is set by (x0, y0) and (x1, y1).

        :param self: self
        :param p: point to reflect (np.array(x, y)) - Point to be reflected
        :param x0: x of the first point of the line - X-coordinate of the first point of the line
        :param y0: y of the first point of the line - Y-coordinate of the first point of the line
        :param x1: x of the second point of the line - X-coordinate of the second point of the line
        :param y1: y of the second point of the line - Y-coordinate of the second point of the line
        :return: edited image - The reflected point as a NumPy array
        """

        dx = x1 - x0
        dy = y1 - y0

        a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
        b = 2 * dx * dy / (dx * dx + dy * dy)

        x2 = int(a * (p[0] - x0) + b * (p[1] - y0) + x0)
        y2 = int(b * (p[0] - x0) - a * (p[1] - y0) + y0)

        return np.array([x2, y2])
