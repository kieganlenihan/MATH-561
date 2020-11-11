
import numpy as np
import copy
import time
from math import sqrt
from scipy.linalg import hilbert
import itertools
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc


tol = 10e-50
def e(n, i):
    v = np.zeros((n, 1))
    v[i] = 1
    return v
def tridiag(A):
    return np.triu(np.tril(A, 1), -1)
def symm(A):
    return (A + A.T)/2
def tridiagonalization(A):
    A = copy.deepcopy(A)
    m = len(A)
    for k in range(m-2):
        x = A[k+1:, k].reshape(m-k-1, 1)
        x_mag = np.linalg.norm(x, 2)
        v = np.sign(x[1]) * x_mag * e(m-k-1, 0) + x
        A[k+1:, k:] -= 2/(np.linalg.norm(v, 2) ** 2) * v @ (v.T @ A[k+1:, k:])
        A[:, k+1:] -= 2/(np.linalg.norm(v, 2) ** 2) * (A[:, k+1:] @ v) @ v.T
    return tridiag(A)
def householder(A):
    R = copy.deepcopy(A)
    m, n = R.shape
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k].reshape(m-k, 1)
        x_mag = np.linalg.norm(x, 2)
        v = np.sign(x[0]) * x_mag * e(m-k, 0) + x
        H = np.eye(m)
        H[k:, k:] = np.eye(m-k)-2 * v @ v.T/(v.T @ v)
        Q = Q @ H
        R[k:, k:] -= 2/(np.linalg.norm(v, 2) ** 2) * v @ (v.T @ R[k:, k:])
    return Q, np.triu(R)
def tri_givens(A):
    R = copy.deepcopy(A)
    m = len(A)
    Q = np.eye(m)
    flag = 1
    for i in range(m-1):
        beta = R[i+1, i]
        if beta == 0:
            pass
        else:
            alpha = R[i, i]
            if abs(beta) >= abs(alpha):
                tau = alpha / beta
                sigma = 1/sqrt(1+tau ** 2)
                gamma = sigma * tau
            else:
                tau = beta / alpha
                gamma = 1/sqrt(1+tau ** 2)
                sigma = gamma * tau
            rotation = np.array([[gamma, sigma], [-sigma, gamma]])
            R[i:i+2, i:i+3] = rotation @ R[i:i+2, i:i+3]
            Q[i:i+2, 0:i+2] = rotation @ Q[i:i+2, 0:i+2]
    return Q.T, np.triu(R)
def qr_check(A):
    q, r = np.linalg.qr(B)
    Q, R = tri_givens(B)
    err = np.max(q @ r - Q @ R)
    if err > tol:
        print('QR ERROR')
def qr_algorithm_unshift(A):
    A = copy.deepcopy(A)
    k = 0
    m = len(A)
    t = []
    ks = []
    while abs(A[m-1, m-2]) > tol:
        k += 1
        Q, R = tri_givens(A)
        A = R @ Q
        t.append(abs(A[m-1, m-2]))
        ks.append(k)
    return ks, t, A
def wilkinson(a, b, c):
    delt = (a-c)/2
    if delt == 0:
        return 1
    return c - np.sign(delt) * b ** 2 / (abs(delt) + sqrt(delt ** 2 + b ** 2))
def qr_algorithm_shift(A, shift):
    A = copy.deepcopy(A)
    k = 0
    m = len(A)
    t = []
    ks = []
    while abs(A[m-1, m-2]) > tol:
        k += 1
        if shift == 'simple':
            mu = A[m-1, m-1]
        elif shift == 'wilkinson':
            mu = wilkinson(A[m-2, m-2], A[m-2, m-1], A[m-1, m-1])
        Q, R = tri_givens(A - mu * np.eye(m))
        A = R @ Q + mu * np.eye(m)
        t.append(abs(A[m-1, m-2]))
        ks.append(k)
    return ks, t, A
def qr_iteration(A, shift):
    A = copy.deepcopy(A)
    m = len(A)
    ts = []
    ks = []
    eigs = []
    max_k = 0
    for i in range(m, 1, -1):
        B = A[0:i, 0:i]
        if len(B) == 1:
            eig = B[0, 0]
        else:
            if shift == 'unshift':
                k, t, A = qr_algorithm_unshift(B)
            else:
                k, t, A = qr_algorithm_shift(B, shift)
            eig = min(np.diag(A))
        eigs.append(eig)
        ts.append(t)
        k = max_k + np.array(k)
        max_k += len(k)
        ks.append(list(k))
    return eigs, ts, ks
def saw_plot(ts, ks, title):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yscale("log")
    plt.xlabel('Iterations (n)')
    plt.ylabel('$|t_{m, m-1}|$')
    n = len(ts[0])
    lst_k = list(itertools.chain(*ks))
    lst_t = list(itertools.chain(*ts))
    plt.plot(lst_k, lst_t)
    for i in range(len(ks)):
        plt.plot(ks[i], ts[i], '*')
    # plt.show()
    plt.savefig(title)
# if __name__ == '__main__':
#     N = 15
#     # B = symm(np.random.rand(N, N))
#     d = sorted([x+1 for x in range(N)], reverse=True)
#     B = np.diag(d) + np.ones(N)
#     B = tridiagonalization(B)
#     # B = tridiagonalization(hilbert(4))
# 
#
#     eigs, ts, ks = qr_iteration(B, 'unshift')
#     saw_plot(ts, ks, 'sawtooth_unshift_big.png')
#     eigs, ts, ks = qr_iteration(B, 'simple')
#     saw_plot(ts, ks, 'sawtooth_simple_shift_big.png')
#     eigs, ts, ks = qr_iteration(B, 'wilkinson')
#     saw_plot(ts, ks, 'sawtooth_wilkinson_shift_big.png')
#     # N = [5, 10, 15, 20, 50, 100]
#     # for i in N:
#     #     d = sorted([x+1 for x in range(i)], reverse=True)
#     #     B = np.diag(d) + np.ones(i)
#     #     B = tridiagonalization(B)
#     #     w, v = np.linalg.eig(B)
#     #     w = sorted(w, reverse=True)
#     #     print(i, w[0])
#     #     _, _, A = qr_algorithm_unshift(B)
#     #     print('unshifted qr err', max(np.diag(A)-w))
#     #     _, _, A = qr_algorithm_unshift(B)
#     #     print('simple qr err', max(np.diag(A)-w))
#     #     _, _, A = qr_algorithm_shift(B, 'wilkinson')
    #     print('wilk qr err', max(np.diag(A)-w))
