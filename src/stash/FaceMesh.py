import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay


class FaceMeshProcessor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def process_image(self, input_image_path, output_image_path):
        input_image = cv2.imread(input_image_path)
        print(input_image)

        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * input_image.shape[1]), int(landmark.y * input_image.shape[0])
                    landmarks.append((x, y))

                points = np.array(landmarks)
                tri = Delaunay(points)

                line_canvas = np.zeros_like(input_image)

                for simplex in tri.simplices:
                    cv2.polylines(line_canvas, [points[simplex]], isClosed=True, color=(0, 255, 0), thickness=1)

                cv2.imwrite(output_image_path, line_canvas)


if __name__ == "__main__":
    processor = FaceMeshProcessor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    processor.process_image('resources/face.jpg', 'output_lines_only.jpg')
