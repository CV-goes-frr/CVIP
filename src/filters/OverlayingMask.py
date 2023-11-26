import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool
import dlib
from imutils import face_utils

from src.filters.Filter import Filter
from settings import prefix
from src.linal.RtcUmeyama import RtcUmeyama

PREDICTOR_PATH = prefix + "src/filters/shape_predictor_81_face_landmarks.dat"


class OverlayingMask(Filter):
    def __init__(self, mask_name: str):
        super().__init__()  # Call the constructor of the parent class (Filter)

        self.detector = dlib.get_frontal_face_detector()  # Initialize the face detector
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Initialize the face landmarks predictor
        self.coef = 1.25

        self.mask_name = mask_name
        self.jaw_mark_l = 1
        self.jaw_mark_r = 15

        self.mask_detection_confidence = 0.5
        self.mask_tracking_confidence = 0.5
        self.mask_max_faces = 1
        # Parameters for mp_face_mesh

        # Specify what landmarks will we use
        self.landmark_points_81 = [127, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63,
                              105, 66, 107, 336,
                              296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362,
                              385, 387, 263, 373,
                              380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14,
                              87, 103, 67, 109, 10, 297, 332, 251, 21, 54, 162, 356, 284, 338]

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        """
        :param self: self
        :param img: np.ndarray of pixels - Input image as a NumPy array
        :param processes_limit:we'll try to parallel it later
        :param pool: processes pool
        :return: edited image - List containing the edited image as a NumPy array
        """

        print("OVERLAYING MASKING IN PROCESS...")

        # Find points on the mask
        mask_image = cv2.imread(self.mask_name)

        # Shape of the mask we need to calculate transformation
        h_mask, w_mask, c_mask = img.shape

        # Change the size of the mask
        mask_image = np.array(self.scale(mask_image, h_mask, w_mask))

        # Find landmarks on the mask
        mp_face_mesh = mp.solutions.face_mesh
        face_points = mp_face_mesh.FaceMesh(max_num_faces=self.mask_max_faces,
                                            refine_landmarks=True,
                                            min_detection_confidence=self.mask_detection_confidence,
                                            min_tracking_confidence=self.mask_tracking_confidence)
        rgb_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_points = face_points.process(rgb_image)

        # get landmarks from FaceMesh class
        for face_landmarks in mask_points.multi_face_landmarks:
            mask_landmarks = []
            for index in self.landmark_points_81:
                x = int(face_landmarks.landmark[index].x * w_mask)
                y = int(face_landmarks.landmark[index].y * h_mask)
                mask_landmarks.append((x, y))

        # Find points on target faces

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        rects = self.detector(gray, 0)  # Detect faces in the grayscale image

        # Process all faces
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)  # Get the facial landmarks for the current face
            shape = face_utils.shape_to_np(shape)  # Convert the landmarks to NumPy array

            R, t, c = RtcUmeyama(shape, mask_landmarks)  # Calculating rotation, translation and scale

            mask_copy = mask_image.copy()

            # Form am Affine transformation matrix
            A = np.concatenate((c * R, np.expand_dims(t, axis=1)), axis=1)
            print("Affine transformation matrix:\n", A)
            mask_copy = cv2.warpAffine(mask_copy, A, (w_mask, h_mask))  # Edit mask image
            # cv2.imwrite("rotated.jpg", mask_copy)

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


if __name__ == "__main__":
    processor = OverlayingMask("Danya.png")
    input_image = cv2.imread("photo.jpg")

    Pool = Pool(processes=2)
    processor.apply(input_image, 2, Pool)
