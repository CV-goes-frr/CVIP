import numpy as np
import dlib
import cv2
from multiprocessing import Pool

from .Filter import Filter

PREDICTOR_PATH = "src/filters/shape_predictor_68_face_landmarks_GTX.dat"

class FaceOilPaint(Filter):
    def __init__(self, coef: str ):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = PREDICTOR_PATH
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.coef = int(coef)

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        faces = self.detector(img)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            face_region = img[y:y+h, x:x+w]

            oil_painted_face = cv2.xphoto.oilPainting(face_region, self.coef, 1)
            img[y:y+h, x:x+w] = oil_painted_face

        return [img]
