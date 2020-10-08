import numpy as np
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
def LHS(m, n):
    ra = np.random.rand(m, n)
    q, r = house(ra)
    A = np.zeros((m+n, m+n))
    A[:m, :m] = np.eye(m)
    A[:m, m:] = ra
    A[m:, :m] = ra.T
    return A, ra
def RHS(m, n):
    b1 = np.random.rand(m, 1)
    b2 = np.random.rand(n, 1)
    return b1, b2
def back_sub(A, b):
    n = b.size
    x = np.zeros_like(b)
    x[n-1] = b[n-1]/A[n-1, n-1]
    U = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        s = 0
        for j in range (i+1, n):
           s += A[i, j]*x[j]
        U[i, i] = b[i] -s
        x[i] = U[i, i]/A[i, i]
    return x
def rSolve(ra, b2):
    L = ra @ ra.T
    qR, rR = house(L)
    R = qR.T @ ra @ b2
    r = back_sub(rR, R)
    return r
def xSolve(ra, b1, r):
    qL, rL = house(ra.T @ ra)
    R = qL.T @ ra.T @ (b1-r)
    x = back_sub(rL, R)
    return x
m = 5
n = 4
A, ra = LHS(m, n)
b1, b2 = RHS(m, n)
r = rSolve(ra, b2)
x = xSolve(ra, b1, r)
r = b1 - ra @ x
x = xSolve(ra, b1, r)
r = b1 - ra @ x
sol = np.vstack((r, x))
b = np.vstack((b1, b2))
# print('true sol', np.linalg.solve(A, b))
# print('my sol', sol)
print('err', np.linalg.norm(np.linalg.solve(A, b)-sol, 2))



# Q, R = house(A)
# x = back_sub(R, Q.T @ b)
# suc = np.linalg.norm(A @ x - b, 1)
# print('b', b)
# print('LHS', R)
# print('RHS', Q.T @ b)
# print('x', x)
