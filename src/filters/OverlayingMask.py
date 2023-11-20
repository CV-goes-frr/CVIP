import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool

from src.filters.Filter import Filter
from src.linal.RtcUmeyama import RtcUmeyama


class Mask(Filter):
    def __init__(self, mask_name: str):
        self.mask_name = mask_name
        super().__init__()  # Call the constructor of the parent class (Filter)

    def apply(self, img: np.ndarray, processes_limit: int):

        # Find points on the mask
        mask_image = cv2.imread(self.mask_name)
        mp_face_mesh = mp.solutions.face_mesh
        face_points = mp_face_mesh.FaceMesh(max_num_faces=20,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        rgb_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_points = face_points.process(rgb_image)

        for face_landmarks in mask_points.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                landmarks.append((x, y))

            mask_landmarks = np.array(landmarks)

        # Find points on target faces
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_faces_points = face_points.process(rgb_image)
        landmarks_all = list()

        if target_faces_points.multi_face_landmarks:
            for face_landmarks in target_faces_points.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    landmarks.append((x, y))

                points = np.array(landmarks)
                landmarks_all.append(points)

        print(len(landmarks_all))
        for landmarks in landmarks_all:
            R, t, c = RtcUmeyama(landmarks, mask_landmarks)
            print("R:", R)
            print("t:", t)
            print("c:", c)
            print()


if __name__ == "__main__":
    processor = Mask("elon.jpg")
    input_image = cv2.imread("face.jpg")
    processor.apply(input_image, 2)
