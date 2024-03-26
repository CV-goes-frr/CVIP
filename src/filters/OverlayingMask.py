from imutils import face_utils
from multiprocessing import Pool

import cv2
import dlib
import numpy as np

from settings import prefix
from .Filter import Filter
from .linal.RtcUmeyama import RtcUmeyama
from ..exceptions.NoFace import NoFaceException


PREDICTOR_PATH = "shape_predictor_81_face_landmarks.dat"


class OverlayingMask(Filter):
    def init(self, mask_name: str):
        """
        Initializes the OverlayingMask filter.

        Args:
            mask_name (str): Name of the mask image file.

        Returns:
            None
        """
        super().init()  # Call the constructor of the parent class (Filter)
        self.log = "OVERLAYING MASKING IN PROGRESS..."

        # Initialize the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        # Coefficient for scaling the mask
        self.coef = 1.25

        # Load the mask image
        self.mask_image = cv2.imread(f'{prefix}/{mask_name}')
        mask_gray = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

        # Detect faces in the mask image
        rects = self.detector(mask_gray, 0)

        # If no faces are detected, raise an exception
        if len(rects) == 0:
            raise NoFaceException(mask_name)

        # Extract facial landmarks from the mask image
        for (i, rect) in enumerate(rects):
            mask_shape = self.predictor(mask_gray, rect)
        self.mask_landmarks = face_utils.shape_to_np(mask_shape)

        # Define indices for marking the jawline on the face

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        """
        Applies the overlaying mask filter to the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            processes_limit (int): Number of processes to use.
            pool (Pool): Pool of processes.

        Returns:
            List[np.ndarray]: List containing the edited image as a NumPy array.
        """
        if self.cache:
            print("USING CACHE...")
            return self.cache

        # Shape of the mask we need to calculate transformation
        h_mask, w_mask, c_mask = img.shape

        # Detect faces in the input image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        rects = self.detector(gray, 0)  # Detect faces in the grayscale image

        # Process all faces
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)  # Get the facial landmarks for the current face
            shape = face_utils.shape_to_np(shape)  # Convert the landmarks to NumPy array

            # Calculate rotation, translation, and scale for overlaying the mask
            R, t, c = RtcUmeyama(shape, self.mask_landmarks)

            mask_copy = self.mask_image.copy()

            # Form an affine transformation matrix
            A = np.concatenate((c * R, np.expand_dims(t, axis=1)), axis=1)
            mask_copy = cv2.warpAffine(mask_copy, A, (w_mask, h_mask))  # Edit mask image

            # Create a silhouette of the face for masking
            face_silhouette = np.concatenate((
                shape[0:17],
                shape[78:79],
                shape[74:75],
                shape[79:80],
                shape[73:74],
                shape[72:73],
                shape[80:81],
                shape[71:72],
                shape[70:71],
                shape[69:70],
                shape[68:69],
                shape[76:77],
                shape[75:76],
                shape[77:78],
                shape[0:1]
            ), axis=0)

            # Create a mask for the face silhouette
            mask_poly = np.zeros_like(mask_copy)
            cv2.fillPoly(mask_poly, [face_silhouette], (255, 255, 255))

            # Overlay the mask on the face silhouette
            img = np.where(mask_poly != 0, mask_copy, img)

        if self.calls_counter > 1:
            self.cache = [img]

        return [img]  # Return the edited image as a list

    @staticmethod
    def scale(im: np.ndarray, nR: np.array, nC: np.array):
        """
        Scales the input image to the specified size.

        Args:
            im (np.ndarray): Input image as a NumPy array.
            nR (np.array): New number of rows.
            nC (np.array): New number of columns.

        Returns:
            np.ndarray: Scaled image as a NumPy array.
        """
        nR0 = len(im)  # Source number of rows
        nC0 = len(im[0])  # Source number of columns
        return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]

    @staticmethod
    def reflect(self, p: np.array, x0: int, y0: int, x1: int, y1: int):
        """
        Reflects a point relative to a line defined by two points.

        Args:
            p (np.array): Point to be reflected.
            x0 (int): X-coordinate of the first point of the line.
            y0 (int): Y-coordinate of the first point of the line.
            x1 (int): X-coordinate of the second point of the line.
            y1 (int): Y-coordinate of the second point of the line.

        Returns:
            np.array: Reflected point as a NumPy array.
        """
        dx = x1 - x0
        dy = y1 - y0

        a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
        b = 2 * dx * dy / (dx * dx + dy * dy)

        x2 = int(a * (p[0] - x0) + b * (p[1] - y0) + x0)
        y2 = int(b * (p[0] - x0) - a * (p[1] - y0) + y0)

        return np.array([x2, y2])
