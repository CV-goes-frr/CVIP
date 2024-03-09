import cv2

image1 = cv2.imread('pic1.png')
image2 = cv2.imread('pic2.png')

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
distance_threshold = 40
good_matches = [match for match in matches if match.distance < distance_threshold]

# Draw only the good matches
matching_result_filtered = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Filtered Feature Matching', matching_result_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
