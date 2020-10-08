import numpy as np
from typing import Union
from mgs import *
from givens import *
import copy
import time

def reflector(a):
    v = a/(a[0]+np.sign(a[0])*np.linalg.norm(a))
    v[0] = 1
    tau = 2 / (v.T @ v)
    return v, tau
def house(A):
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for k in range(n):
        v, tau = reflector(R[k:, k, np.newaxis])
        H = np.identity(m)
        H[k:, k:] -= tau * (v @ v.T)
        R = H @ R
        Q = H @ Q
    return Q[:n].T, np.triu(R[:n])
# m = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # n = 40
# ho, gi = [], []
# for r in m:
#     A,_,_ = mat0(r, r, 2)
#     start = time.time()
#     Q, R = house(A)
#     ho.append(time.time()-start)
#     start = time.time()
#     Q, R = givens_QR(A)
#     gi.append(time.time()-start)
# import matplotlib.pyplot as plt
# plt.yscale('log')
# leg = ['Householder', 'Givens']
# plt.plot(m, ho, 'ro')
# plt.plot(m, gi, 'bo')
# plt.legend(leg, loc = 'best')
# plt.grid()
# plt.xlabel('n')
# plt.ylabel('Time (s)')
# plt.savefig('time_giv_ho.png')
# plt.show()

# print('r', R)
# # print(R)
# # Q, R = qr2(A)
# print('orth norm', np.linalg.norm(Q.T @ Q - np.identity(n), 2))
# print('true orth norm', np.linalg.norm(q.T @ q - np.identity(n), 2))
# print('ground norm', np.linalg.norm(Q @ R - A, 2))
