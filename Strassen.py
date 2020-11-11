import numpy as np
import time
import matplotlib.pyplot as plt
from math import log

def split(A):
    n = len(A)
    if n % 2 == 0:
        m = int(n/2)
    else:
        m = int(n/2+.5)
    return A[:m, :m], A[:m, m:], A[m:, :m], A[m:, m:]
def strassen(M, N):
    n = len(M)
    A, B, C, D = split(M)
    E, F, G, H = split(N)
    start = time.time()
    P1 = (A+D) @ (E+H)
    P2 = (C+D) @ E
    P3 = A @ (F-H)
    P4 = D @ (G-E)
    P5 = (A+B) @ H
    P6 = (C-A) @ (E+F)
    P7 = (B-D) @ (G+H)
    W = P1+P4-P5+P7
    X = P3+P5
    Y = P2+P4
    Z = P1+P3-P2+P6
    return time.time()-start, np.vstack((np.hstack((W, X)), np.hstack((Y, Z))))
def naive(M, N):
    n = len(M)
    A, B, C, D = split(M)
    E, F, G, H = split(N)
    return np.vstack((np.hstack((A @ E+B @ G, A @ F+B @ H)), np.hstack((C @ E+D @ G, C @ F+D @ H))))
if __name__ == '__main__':
    N = 11
    ms, ts = [], []
    plt.yscale("log")
    plt.xscale("log")
    for k in range(N):
        print(k)
        m = 2 ** (k+1)
        A  = np.random.randint(-5, 5, size = (m, m))
        B = np.random.randint(-5, 5, size = (m, m))
        stras_t, _ = strassen(A, B)
        plt.plot(m, stras_t, 'k.')
        plt.plot(m, m ** log(7, 2)* 10 ** -8, 'b.')
        ms.append(m)
        ts.append(stras_t)
    plt.plot(ms, ts, 'k', label = "Strassen")
    plt.plot(ms, [m ** log(7, 2)* 10 ** -8 for m in ms], 'b', label = "$O(m^{\log_2(7)})$")
    plt.ylabel("Calculation Time (s)")
    plt.xlabel("Matrix Dimension $m$")
    plt.legend()
    # plt.show()
    plt.savefig("Strassen.png")
