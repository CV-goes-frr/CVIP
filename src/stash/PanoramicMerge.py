import cv2
import numpy as np


def main():
    imageR = cv2.imread("test2.png")
    imageL = cv2.imread("test1.png")
    homography = get_homography(imageR, imageL)

    if homography is None:
        return

    result = cv2.warpPerspective(imageR, homography,
                                 (imageR.shape[1] + imageL.shape[1], imageR.shape[0]))
    result[0:imageL.shape[0], 0:imageL.shape[1]] = imageL

    cv2.imwrite("panorama.png", result)


def get_homography(image1: np.ndarray, image2: np.ndarray):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

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
        print(H)
        return H
    else:
        print("Not enough feature matches to create a panorama")
        return None


if __name__ == "__main__":
    main()
