import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
import copy

def mat_construction_prob1(m, ex):
    A = 2*np.eye(m) + np.random.normal(0, .5, size = (m, m))/math.sqrt(m)
    if ex == 1:
        return A
    else:
        A = A.astype(np.complex64)
        for i in range(m):
            theta = (i * math.pi)/(m-1)
            A[i, i] = complex(2*math.sin(theta), math.cos(theta))
    return A
def prob1(ex):
    m = 200
    A = mat_construction_prob1(m, ex)
    w, v = np.linalg.eig(A)
    fg, axs = plt.subplots()
    plt.plot(w.real, w.imag, 'o', label = 'Eigenvalues')
    if ex == 1:
        circ = plt.Circle((2, 0), .5, fill=False)
        axs.add_patch(circ)
    plt.legend()
    plt.xlabel('Real Axis $\mathbb{R}$')
    plt.ylabel('Complex Axis $\mathbb{C}$')
    plt.axis('equal')
    plt.xlim(0, 3)
    plt.ylim(-1.5, 1.5)
    plt.show()
    plt.savefig('disk_eig_hollow.png')
    plt.yscale('log')
    b = np.ones((m, 1))
    res, cnt = [], []
    class gmres_cnt(object):
        def __init__(self):
            self.n = 0
        def __call__(self, rk=None):
            self.n += 1
            res.append(rk)
            cnt.append(self.n)
    counter = gmres_cnt()
    x, info = gmres(A, b, callback=counter)
    plt.plot(cnt, res, 'o')
    fr = []
    cnt.insert(0, 0)
    cnt.insert(counter.n+1, counter.n+1)
    for i in cnt:
        fr.append(4 ** -i)
    if ex == 1:
        plt.plot(cnt, fr, 'k--', label = '$4^{-n}$')
    plt.xlabel('Iterations $n$')
    h = plt.ylabel('$\dfrac{||r_n||}{||b||}$       ')
    h.set_rotation(0)
    plt.xlim(0, 12)
    plt.ylim(10 ** -6, 10 ** 1)
    plt.show()
    # plt.savefig('gmres_cnv_hollow.png')
def CG(A, b, x):
    r = copy.deepcopy(b)
    p = copy.deepcopy(b)
    res, rn, ns = [], [], []
    for n in range(len(A)):
        alpha = ((r.T @ r) / (p.T @ A @ p)).item()
        x = x + alpha * p
        r_ = r - alpha * A @ p
        beta = ((r_.T @ r_)/ (r.T @ r)).item()
        p_ = r_+ beta * p
        p = p_
        r = r_
        rn.append(np.linalg.norm(r_, 2))
        res.append(np.linalg.norm(b - A @ x, 2))
        ns.append(n)
    return rn, res, ns
def mat_construction_prob2(m):
    main_diag = [x for x in range(m)]
    sec_diag = [1 for x in range(m-1)]
    return np.diag(main_diag) + np.diag(sec_diag, k=-1) + np.diag(sec_diag, k=1)
def prob2():
    m = 100
    A = mat_construction_prob2(m)
    b = np.ones((m, 1))
    x0 = np.zeros((m, 1))
    rn, res, ns = CG(A, b, x0)
    plt.yscale('log')
    plt.plot(ns, rn, '.', label = '$||r_n||_2$')
    plt.plot(ns, res, '.', label = '$||b-Ax_n||_2$')
    plt.xlabel('Iterations $n$')
    plt.ylabel('$r_n')
    plt.legend()
    h = plt.ylabel('$r_n$')
    h.set_rotation(0)
    # plt.show()
    plt.savefig('CG_tridiag.png')
def GMRES
if __name__ == '__main__':
    # prob1(1)
    prob2()
