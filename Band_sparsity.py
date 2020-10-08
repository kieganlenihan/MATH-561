from scipy.sparse import diags
import numpy as np

# diagonals = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3], [1, 2]]
# x = diags(diagonals, [0, -1, 1, -2, -3, -4]).toarray()
x = np.array([[-1, -1, 1], [1, 3, 3], [-1, -1, 5]])
q, r = np.linalg.qr(x)
L = np.linalg.cholesky(x)
print(r*-1)
print(L)
