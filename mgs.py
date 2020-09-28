import numpy as np
from scipy.stats import ortho_group
from clgs import *
import copy

def mgs(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = copy.deepcopy(A)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:,i], 2)
        Q[:, i] = V[:,i]/R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.matmul(Q[:, i].T, V[:,j])
            V[:, j] = V[:, j]-R[i, j]*Q[:, i]
    return Q, R
def mat0(m, n):
    rand_orth = ortho_group.rvs(m)
    Q0 = rand_orth[:,:n]
    rand_norm = np.random.normal(0, .1, size=(np.sum(np.arange(1, n)), 1))
    R = np.zeros((n, n))
    k=0
    for j in range(n):
        for i in range(n):
            if j == i:
                R[j, j] = 1
            elif j > i:
                R[i, j] = rand_norm[k]
                k += 1
    diag_list = []
    for i in range(n):
        diag_list.append(2**(-i-1))
    S = np.diagflat([diag_list])
    R0 = np.matmul(S, R)
    return np.matmul(Q0, R0), Q0, R0
def error(m, n):
    A, Q0, R0 = mat0(m, n)
    Q, R = mgs(A)
    Q1, R1 = clgs(A)
    Q2, R2 = np.linalg.qr(A)
    qs = [Q, Q1, Q2]
    rs = [R, R1, R2]
    orth_error, ground_error = [], []
    for q, r in zip(qs, rs):
        mat = np.matmul(q.T, q)-np.identity(n)
        orth_error.append(np.linalg.norm(mat, 2))
        mat2 = np.matmul(q, r)-A
        ground_error.append(np.linalg.norm(mat2, 2))
    return orth_error, ground_error

m = 40
n = 40
arr = np.empty([0, 3])
arr2 = np.empty([0, 3])
for i in range(20):
    orth_error, ground_error = error(m, n)
    arr = np.vstack((arr, orth_error))
    arr2 = np.vstack((arr2, ground_error))
orth_mean_mgs = np.mean(arr[:,0])
orth_mean_clgs = np.mean(arr[:,1])
orth_mean_std = np.mean(arr[:,2])
orth_std_mgs = np.std(arr[:,0])
orth_std_clgs = np.std(arr[:,1])
orth_std_std = np.std(arr[:,2])
print(orth_mean_mgs, orth_mean_clgs, orth_mean_std)
print(orth_std_mgs, orth_std_clgs, orth_std_std)
ground_mean_mgs = np.mean(arr2[:, 0])
ground_mean_clgs = np.mean(arr2[:, 1])
ground_mean_std = np.mean(arr2[:, 2])
ground_std_mgs = np.std(arr2[:, 0])
ground_std_clgs = np.std(arr2[:, 1])
ground_std_std = np.std(arr2[:, 2])
print('')
print(ground_mean_mgs, ground_mean_clgs, ground_mean_std)
print(ground_std_mgs, ground_std_clgs, ground_std_std)
