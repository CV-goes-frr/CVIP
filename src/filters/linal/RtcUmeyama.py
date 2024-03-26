import numpy as np


def RtcUmeyama(A: np.ndarray, B: np.ndarray) -> tuple:
    """
    Estimate the rigid transformation between two sets of 3D points using the Umeyama algorithm.

    Args:
        A (np.ndarray): Source point set, shape (n, 3).
        B (np.ndarray): Target point set, shape (n, 3).

    Returns:
        tuple: Rotation matrix R, translation vector t, scaling factor c.
    """

    n, m = A.shape  # Get the number of points and dimensions

    centroid_A = np.mean(A, axis=0)  # Calculate the centroid of A
    centroid_B = np.mean(B, axis=0)  # Calculate the centroid of B

    # Compute the variance of A
    variance = np.mean(np.linalg.norm(A - centroid_A, axis=1) ** 2)

    # Compute the cross-covariance matrix H
    H = (np.dot((A - centroid_A).T, (B - centroid_B))) / n

    # Perform singular value decomposition of H
    U, D, VT = np.linalg.svd(H)

    # Determine the sign of the determinant of U*VT
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))

    # Construct the scaling matrix S
    S = np.diag([1] * (m - 1) + [d])

    # Compute the rotation matrix R
    R = np.dot(U, np.dot(S, VT))

    # Compute the scaling factor c
    c = variance / np.trace(np.diag(D) @ S)

    # Compute the translation vector t
    t = centroid_A - c * np.dot(R, centroid_B)

    return R, t, c
