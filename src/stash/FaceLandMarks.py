from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from multiprocessing import Pool

from src.filters.Filter import Filter

MODEL_PATH = 'src/filters/face_landmarker.task'


class FaceLandMarks(Filter):
    def __init__(self):
        super().__init__()  # Call the constructor of the parent class (Filter)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        result_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=result_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=result_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=result_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return result_image

    def apply(self, img: np.ndarray, processes_limit: int, pool: Pool):
        """
            Face detection with cv2.CascadeClassifier().
            :param img: np.ndarray of pixels - Input image as a NumPy array
            :param processes_limit: we'll try to parallel it later
            :param pool: processes pool
            :return: List containing the edited image as a NumPy array
        """

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=img)
        detection_result = detector.detect(mp_image)
        result_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        return [result_image]
