import cv2
import numpy as np
import mediapipe as mp
from multiprocessing import Pool

from src.filters.BilinearScale import BilinearScale
from src.filters.Filter import Filter
from src.linal.RtcUmeyama import RtcUmeyama


class Mask(Filter):
    def __init__(self, mask_name: str):
        self.mask_name = mask_name
        super().__init__()  # Call the constructor of the parent class (Filter)

    @staticmethod
    def scale(im, nR, nC):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns
        return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        # Find points on the mask
        mask_image = cv2.imread(self.mask_name)

        # Shape of the mask we need to calculate transformation
        h_mask, w_mask, c_mask = img.shape

        # Change the size of the mask
        mask_image = np.array(self.scale(mask_image, h_mask, w_mask))
        print(mask_image.shape)

        # Find landmarks on the mask
        mp_face_mesh = mp.solutions.face_mesh
        face_points = mp_face_mesh.FaceMesh(max_num_faces=20,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        rgb_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_points = face_points.process(rgb_image)

        # get landmarks from FaceMesh class
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
                landmarks_all.append(points)  # Get landmarks to the list

        print(len(landmarks_all))

        # Process all faces
        for landmarks in landmarks_all:
            R, t, c = RtcUmeyama(landmarks, mask_landmarks)  # Calculating rotation, translation and scale
            print("R:", R)
            print("t:", t)
            print("c:", c)
            print()
            mask_copy = mask_image.copy()

            # Form am Affine transformation matrix
            A = c * np.concatenate((R, np.expand_dims(t, axis=1)), axis=1)
            print("Affine transformation matrix:\n", A)
            mask_copy = cv2.warpAffine(mask_copy, A, (w_mask, h_mask))  # Edit mask image
            cv2.imwrite("rotated.jpg", mask_copy)

            # Create a poly on the face, where we'll change pixels to mask's pixels
            mask_poly = np.zeros_like(mask_copy)
            cv2.fillPoly(mask_poly, [landmarks], (255, 255, 255))
            img = np.where(mask_poly != 0, mask_copy, img)
            cv2.imwrite("result.jpg", img)


if __name__ == "__main__":
    processor = Mask("elon.jpg")
    input_image = cv2.imread("face.jpg")

    Pool = Pool(processes=2)
    processor.apply(input_image, 2, Pool)
