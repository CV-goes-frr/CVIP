import unittest

import cv2
import numpy as np
from multiprocessing import Pool

from settings import prefix
from src.filters.BilinearScale import BilinearScale, weight_function
from src.filters.FeatureMatching import FeatureMatching
from src.filters.MotionTracking import MotionTracking
from src.filters.OverlayingMask import OverlayingMask
from src.filters.ScaleToResolution import ScaleToResolution
from src.filters.Crop import Crop
from src.filters.NnScale import NnScale
from src.filters.FaceBlurrer import FaceBlurrer
from src.filters.FaceDetection import FaceDetection


class TestBicubicScale(unittest.TestCase):

    def test_bicubic_scale(self):
        test_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
            [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
        ], dtype=np.uint8)

        scale_factor = 2.0
        bicubic_filter = ScaleToResolution(scale_factor)

        with Pool(processes=2) as pool:
            scaled_image = bicubic_filter.apply(test_image, 2, pool)[0]

        self.assertEqual((6, 6, 3), scaled_image.shape)
        expected_values = [
            [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)],
            [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)],
            [(0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255), (255, 255, 0), (255, 255, 0)],
            [(0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255), (255, 255, 0), (255, 255, 0)],
            [(128, 128, 128), (128, 128, 128), (64, 64, 64), (64, 64, 64), (192, 192, 192), (192, 192, 192)],
            [(128, 128, 128), (128, 128, 128), (64, 64, 64), (64, 64, 64), (192, 192, 192), (192, 192, 192)]
        ]

        for y in range(6):
            for x in range(6):
                self.assertEqual(expected_values[y][x], tuple(scaled_image[y, x]))

    def test_weight_function(self):
        """
        ((1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right
            + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right) =
            ((1 - 0.5) * (1 - 0.5) * [100, 50, 25] + 0.5 * (1 - 0.5) * [150, 75, 38]
            + (1 - 0.5) * 0.5 * [75, 38, 19] + 0.5 * 0.5 * [125, 63, 31]) =
            (0.25 * [100, 50, 25] + 0.25 * [150, 75, 38] + 0.25 * [75, 38, 19] +
            0.25 * [125, 63, 31]) = [25, 12.5, 6.25] + [37.5, 18.75, 9.5] +
            [18.75, 9.5, 4.75] + [31.25, 15.75, 7.75] = [112.5, 56.5, 28.5] = [112, 56, 28]
        """
        test_cases = [
            {
                'alpha': 0.5,
                'beta': 0.5,
                'top_left': np.array([100, 50, 25], dtype=np.uint8),
                'top_right': np.array([150, 75, 38], dtype=np.uint8),
                'bottom_left': np.array([75, 38, 19], dtype=np.uint8),
                'bottom_right': np.array([125, 63, 31], dtype=np.uint8),
                'expected_result': np.array([112, 56, 28], dtype=np.uint8),
            }
        ]

        for test_case in test_cases:
            alpha = test_case['alpha']
            beta = test_case['beta']
            top_left = test_case['top_left']
            top_right = test_case['top_right']
            bottom_left = test_case['bottom_left']
            bottom_right = test_case['bottom_right']
            expected_result = test_case['expected_result']

            result = weight_function(alpha, beta, top_left, top_right, bottom_left, bottom_right)

            self.assertTrue(np.array_equal(expected_result, result))

    def test_bilinear_scale(self):
        test_image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[128, 128, 128], [192, 192, 192]]
        ], dtype=np.uint8)

        scale_factor = 2.0
        bilinear_filter = BilinearScale(scale_factor)

        with Pool(processes=2) as pool:
            scaled_image = bilinear_filter.apply(test_image, 2, pool)[0]

        expected_result = np.array([
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[128, 128, 128], [128, 128, 128], [192, 192, 192], [192, 192, 192]],
            [[128, 128, 128], [128, 128, 128], [192, 192, 192], [192, 192, 192]]
        ], dtype=np.uint8)

        self.assertEqual((4, 4, 3), scaled_image.shape)
        self.assertTrue(np.array_equal(expected_result, scaled_image))

    def test_crop(self):
        test_image = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
            [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42], [43, 44, 45]],
            [[46, 47, 48], [49, 50, 51], [52, 53, 54], [55, 56, 57], [58, 59, 60]],
            [[61, 62, 63], [64, 65, 66], [67, 68, 69], [70, 71, 72], [73, 74, 75]],
        ], dtype=np.uint8)

        x1, y1, x2, y2 = 1, 1, 4, 4

        crop_filter = Crop(x1, y1, x2, y2)

        with Pool(processes=1) as pool:
            cropped_image = crop_filter.apply(test_image, 1, pool)[0]

        expected_result = np.array([
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            [[34, 35, 36], [37, 38, 39], [40, 41, 42]],
            [[49, 50, 51], [52, 53, 54], [55, 56, 57]],
        ], dtype=np.uint8)

        self.assertEqual((3, 3, 3), cropped_image.shape)
        self.assertTrue(np.array_equal(expected_result, cropped_image))

    def test_crop_exception(self):
        test_image = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
            [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42], [43, 44, 45]],
            [[46, 47, 48], [49, 50, 51], [52, 53, 54], [55, 56, 57], [58, 59, 60]],
            [[61, 62, 63], [64, 65, 66], [67, 68, 69], [70, 71, 72], [73, 74, 75]],
        ], dtype=np.uint8)

        x1, y1, x2, y2 = 4, 4, 1, 1

        crop_filter = Crop(x1, y1, x2, y2)

        with Pool(processes=1) as pool:
            with self.assertRaises(Exception):
                crop_filter.apply(test_image, 1, pool)

    def test_nn_scale(self):
        test_image = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[128, 128, 128], [192, 192, 192]]
        ], dtype=np.uint8)

        scale_factor = 1.5
        nn_filter = NnScale(scale_factor)

        with Pool(processes=2) as pool:
            scaled_image = nn_filter.apply(test_image, 2, pool)[0]

        expected_result = np.array([
            [[255, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[255, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[128, 128, 128], [128, 128, 128], [192, 192, 192]]
        ], dtype=np.uint8)

        self.assertEqual((3, 3, 3), scaled_image.shape)
        self.assertTrue(np.array_equal(expected_result, scaled_image))

    def test_face_blurrer_not_face(self):
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        blurrer = FaceBlurrer(coef='30')

        with Pool(processes=2) as pool:
            blurred_images = blurrer.apply(test_image, 2, pool)
            blurred_image = blurred_images[0]

        self.assertTrue(np.array_equal(test_image, blurred_image))  # nothing has changed

    def test_face_blurrer_face(self):
        test_image = cv2.imread(prefix + 'resources/face.jpg')

        blurrer = FaceBlurrer(coef='30')

        with Pool(processes=2) as pool:
            blurred_images = blurrer.apply(test_image, 2, pool)
            blurred_image = blurred_images[0]

        self.assertFalse(np.array_equal(test_image, blurred_image))  # result image is different

    def test_face_detection_not_face(self):
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        detection = FaceDetection()

        with Pool(processes=2) as pool:
            detected_image = detection.apply(test_image, 2, pool)[0]

        self.assertTrue(np.array_equal(test_image, detected_image))  # nothing has changed

    def test_face_detection_face(self):
        test_image = cv2.imread(prefix + 'resources/face.jpg')

        detection = FaceDetection()

        with Pool(processes=2) as pool:
            detected_image = detection.apply(test_image, 2, pool)[0]

        self.assertFalse(np.array_equal(test_image, detected_image))  # face detected

    def test_overlaying_mask_face(self):
        test_image = cv2.imread(prefix + 'resources/face.jpg')

        overlaying_mask = OverlayingMask(mask_name=prefix + 'resources/elon.jpg')

        with Pool(processes=2) as pool:
            masked_image = overlaying_mask.apply(test_image, 2, pool)[0]

        self.assertFalse(np.array_equal(test_image, masked_image))  # overlaying mask

    def test_overlaying_mask_without_face(self):
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        overlaying_mask = OverlayingMask(mask_name=prefix + 'resources/elon.jpg')

        with Pool(processes=2) as pool:
            masked_image = overlaying_mask.apply(test_image, 2, pool)[0]

        self.assertTrue(np.array_equal(test_image, masked_image))  # nothing has changed

    def test_feature_matching_any_types(self):
        test_image = cv2.imread(prefix + "resources/book1.jpg")

        feature_matching_bf = FeatureMatching(type_match='BF', img2=prefix + "resources/book2.jpg")
        feature_matching_flann = FeatureMatching(type_match='FLANN', img2=prefix + "resources/book2.jpg")

        with Pool(processes=2) as pool:
            matched_bf = feature_matching_bf.apply(test_image, 2, pool)[0]
            matched_flann = feature_matching_flann.apply(test_image, 2, pool)[0]

        self.assertFalse(np.array_equal(matched_flann, matched_bf))

    def test_feature_matching_bf(self):
        test_image = cv2.imread(prefix + "resources/book1.jpg")

        feature_matching_bf = FeatureMatching(type_match='BF', img2=prefix + "resources/book2.jpg")

        with Pool(processes=2) as pool:
            matched_bf = feature_matching_bf.apply(test_image, 2, pool)[0]

        self.assertFalse(np.array_equal(test_image, matched_bf))

    def test_feature_matching_flann(self):
        test_image = cv2.imread(prefix + "resources/book1.jpg")

        feature_matching_flann = FeatureMatching(type_match='FLANN', img2=prefix + "resources/book2.jpg")

        with Pool(processes=2) as pool:
            matched_bf = feature_matching_flann.apply(test_image, 2, pool)[0]

        self.assertFalse(np.array_equal(test_image, matched_bf))

    def test_motion_tracking_true(self):
        cap = cv2.VideoCapture(prefix + "resources/Patrick.mp4")
        ret, first_frame = cap.read()  # reading first frame
        ret, second_frame = cap.read()  # reading second frame

        motion_tracking = MotionTracking()

        with Pool(processes=2) as pool:
            edited_frame = motion_tracking.apply(first_frame, second_frame, 2, pool)[0]

        self.assertFalse(np.array_equal(edited_frame, first_frame))

    def test_motion_tracking_false(self):
        cap = cv2.VideoCapture(prefix + "resources/Patrick.mp4")
        ret, first_frame = cap.read()  # reading frame

        motion_tracking = MotionTracking()

        with Pool(processes=2) as pool:
            edited_frame = motion_tracking.apply(first_frame, first_frame, 2, pool)[0]

        self.assertTrue(np.array_equal(edited_frame, first_frame))


if __name__ == '__main__':
    unittest.main()
