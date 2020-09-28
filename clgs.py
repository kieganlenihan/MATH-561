import numpy as np

def clgs(A):
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))
    R[0, 0] = np.linalg.norm(A[:, 0], 2)
    Q[:, 0] = A[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
        z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = np.linalg.norm(z, 2)
        Q[:m, k] = z / R[k, k]
    return Q, R
