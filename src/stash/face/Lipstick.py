import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor_path = "../../filters/shape_predictor_81_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmark_points = []
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))


        lip_points = landmark_points[48:60] + landmark_points[60:68]

        lip_pixels = np.array(lip_points)
        green_color = (32, 21, 191)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.int32(lip_pixels)], green_color)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        left_eyebrow_points = landmark_points[17:22]
        right_eyebrow_points = landmark_points[22:27]

        eyebrow_pixels_left = np.array(left_eyebrow_points)
        eyebrow_pixels_right = np.array(right_eyebrow_points)

        blue_color = (3, 3, 56)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.int32(eyebrow_pixels_left)], blue_color)
        cv2.fillPoly(overlay, [np.int32(eyebrow_pixels_right)], blue_color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        left_eye_points = landmark_points[36:42]
        right_eye_points = landmark_points[42:48]

        left_eye_pixels = np.array(left_eye_points)
        right_eye_pixels = np.array(right_eye_points)

        cv2.polylines(frame, [left_eye_pixels], isClosed=True, color=(0, 0, 0), thickness=2)
        cv2.polylines(frame, [right_eye_pixels], isClosed=True, color=(0, 0, 0), thickness=2)

    cv2.imshow('Colored Lip and Eyebrow Regions', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
