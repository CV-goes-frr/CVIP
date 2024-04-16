import cv2
import numpy as np

image1 = cv2.imread("./media/one.jpg")
image2 = cv2.imread("./media/two.jpg")

# Apply GaussianBlur to reduce noise
gray1 = cv2.GaussianBlur(image1, (5, 5), 0)
gray2 = cv2.GaussianBlur(image2, (5, 5), 0)

# Convert images to grayscale
gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

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

minMatches = 10
if len(good_matches) > minMatches:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w, _ = image1.shape
    object_bbox = cv2.boundingRect(np.intp(src_pts))

    pts = np.float32([[[object_bbox[0], object_bbox[1]],
                      [object_bbox[0], object_bbox[1] + object_bbox[3]],
                      [object_bbox[0] + object_bbox[2],
                       object_bbox[1] + object_bbox[3]],
                      [object_bbox[0] + object_bbox[2], object_bbox[1]]]]).reshape(-1, 1, 2)
    dst = np.int32(cv2.perspectiveTransform(pts, M))
    image2 = cv2.polylines(image2, [dst], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough features")
# Draw only the good matches
matching_result_filtered = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("result.jpg", matching_result_filtered)
