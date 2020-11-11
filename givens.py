import numpy as np
import math
import copy

def givensrotation(a, b):
    if b == 0:
        c = 1
        s = 1
    elif abs(b) > abs(a):
        r = a/b
        s = 1/math.sqrt(1+r**2)
        c = s*r
    else:
        r = b/a
        c = 1/math.sqrt(1+r**2)
        s = c*r
    return c, s
def givens_QR(A):
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for j in range(n):
        for i in range(m-1, j, -1):
            G = np.identity(m)
            c, s = givensrotation(R[i-1, j], R[i, j])
            G[i-1:i+1, i-1:i+1] = np.array([[c, -s], [s, c]])
            R = G.T @ R
            Q = Q @ G
    return Q, R
