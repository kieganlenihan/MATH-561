import numpy as np
from scipy import linalg

def cgs(A):
    """Classical Gram-Schmidt (CGS) algorithm"""
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))
    R[0, 0] = linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
        z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = linalg.norm(z) ** 2
        Q[:m, k] = z / R[k, k]
    return Q, R

# m = 6
n = 80
# X = np.random.random((m, n))]
[U, _] = cgs(np.random.random((n, n)))
[V, _] = cgs(np.random.random((n, n)))
diag_list = []
for i in range(80):
    diag_list.append(2**(-i-1))
S = np.diagflat([diag_list])
X = S
# sig = 10**-11
# X = np.array([[10, 15], [10+sig, 15]])
# X = np.array([[.70000, .70711], [.70001, .70711]])
# print('X', X)
Q, R = cgs(X)
print('Q', Q)
print('R', R)
mat = np.matmul(Q.T, Q)-np.identity(n)
print('2 norm', np.linalg.norm(mat, 2))
print('fro norm', np.linalg.norm(mat, 'fro'))
