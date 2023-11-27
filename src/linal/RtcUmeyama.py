import numpy as np


def RtcUmeyama(A: np.ndarray, B: np.ndarray) -> tuple:
    n, m = A.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    variance = np.mean(np.linalg.norm(A - centroid_A, axis=1) ** 2)

    H = (np.dot((A - centroid_A).T, (B - centroid_B))) / n

    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = np.dot(U, np.dot(S, VT))
    c = variance / np.trace(np.diag(D) @ S)
    t = centroid_A - c * np.dot(R, centroid_B)

    return R, t, c
