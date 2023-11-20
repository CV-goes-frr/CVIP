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


def main():
    A = np.array([[23, 178],
                  [66, 173],
                  [88, 187],
                  [119, 202],
                  [122, 229],
                  [170, 232],
                  [179, 199]])
    B = np.array([[232, 38],
                  [208, 32],
                  [181, 31],
                  [155, 45],
                  [142, 33],
                  [121, 59],
                  [139, 69]])

    R, t, c = RtcUmeyama(A, B)
    print("R: ", R)
    print("t: ", t)
    print("c: ", c)


if __name__ == "__main__":
    main()
