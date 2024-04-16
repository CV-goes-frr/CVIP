import cv2
import numpy as np


class PanoramicMerge:
    @staticmethod
    def process(imageL: np.ndarray, imageR: np.ndarray):
        homography = PanoramicMerge.get_homography(imageR, imageL)

        if homography is None:
            print('Cannot calculate homography')
            return

        warped = cv2.warpPerspective(imageR, homography,
                                     (imageR.shape[1] + imageL.shape[1], imageL.shape[0]))
        mask_warped = np.sum(warped, axis=2) != 0
        mask_warped = cv2.erode(mask_warped.astype(np.uint8), np.ones((3, 3)))  # delete edge points and artifacts

        result = np.ndarray((imageL.shape[0], imageL.shape[1] + imageR.shape[1], 3))

        result[0:imageL.shape[0], 0:imageL.shape[1]] = imageL  # left part
        result[mask_warped != 0] = warped[mask_warped != 0]  # right part where are no left part pixels

        result = result.astype(np.uint8)

        # remove black space at the right
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        threshold, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            result = result[y:y + h, x:x + w]

        return result

    @staticmethod
    def get_homography(image1: np.ndarray, image2: np.ndarray):
        # print(image1.shape)
        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 2)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 2)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        # Use a feature matcher to find matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches based on their distances
        matches = sorted(matches, key=lambda x: x.distance)

        # Set a distance threshold
        distance_threshold = 100
        good_matches = [match for match in matches if match.distance < distance_threshold]

        # Find the Homography matrix
        minMatches = 10
        if len(good_matches) > minMatches:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # print(H)
            return H
        else:
            print("Not enough feature matches to create a panorama")
            return None


def main():
    imageR = cv2.imread("../media/test2.png")
    imageL = cv2.imread("../media/test1.png")
    cv2.imwrite("../media/panorama.png", PanoramicMerge.process(imageL, imageR))


if __name__ == "__main__":
    main()
