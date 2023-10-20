import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import Delaunay

# Load the shape predictor model for facial landmarks
PREDICTOR_PATH = "shape_predictor_68_face_landmarks_GTX.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Function to find 68 facial landmarks in an image
def find_facial_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    landmarks = face_utils.shape_to_np(shape)
    return landmarks

# Function to calculate weighted average of points for morphing
def calculate_morphed_points(points1, points2, alpha):
    return (1 - alpha) * points1 + alpha * points2

# Function to calculate the affine transformation matrix between two triangles
def calculate_affine_transform(triangle1, triangle2):
    matrix = cv2.getAffineTransform(triangle1, triangle2)
    return matrix

# Function to warp and blend a triangle from both input images
def warp_triangle(image1, image2, triangle1, triangle2, alpha):
    # Create empty masks for both triangles
    mask1 = np.zeros_like(image1)
    mask2 = np.zeros_like(image2)

    # Fill the masks with the triangle
    cv2.fillConvexPoly(mask1, triangle1, (1, 1, 1))
    cv2.fillConvexPoly(mask2, triangle2, (1, 1, 1))

    # Warp the triangles using the affine transformation
    warp1 = cv2.warpAffine(image1, calculate_affine_transform(triangle1, triangle2), (image1.shape[1], image1.shape[0]))
    warp2 = cv2.warpAffine(image2, calculate_affine_transform(triangle2, triangle1), (image2.shape[1], image2.shape[0]))

    # Blend the warped triangles based on alpha
    result_triangle = (1 - alpha) * warp1 + alpha * warp2

    # Multiply the result triangle by the mask to remove background
    result_triangle *= mask1

    return result_triangle

def main():
    # Load input images
    image1 = cv2.imread("face.jpg")
    image2 = cv2.imread("face2.jpg")

    # Find 68 facial landmarks for both images
    landmarks1 = find_facial_landmarks(image1)
    landmarks2 = find_facial_landmarks(image2)

    if landmarks1 is None or landmarks2 is None:
        print("Face not found in one of the images.")
        return

    # Create Delaunay triangulation for both sets of landmarks
    tri1 = Delaunay(landmarks1)
    tri2 = Delaunay(landmarks2)

    # Create an output (morphed) image
    morphed_image = np.zeros_like(image1)

    # Set alpha value between 0 and 1
    alpha = 0.5

    # Iterate through triangles and perform warp and blend
    for triangle1, triangle2 in zip(tri1.simplices, tri2.simplices):
        # Calculate morphed points
        morphed_points = calculate_morphed_points(landmarks1[triangle1], landmarks2[triangle2], alpha)

        # Warp and blend the triangle
        warped_triangle = warp_triangle(image1, image2, landmarks1[triangle1], morphed_points, alpha)

        # Add the warped triangle to the morphed image
        morphed_image += warped_triangle

    # Display or save the morphed image
    cv2.imshow("Morphed Image", morphed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
