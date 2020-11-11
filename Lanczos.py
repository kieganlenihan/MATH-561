import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.ticker import MaxNLocator
from cycler import cycler

def lanc_matrix():
    v = []
    for i in range(1, 200):
        v.append(i*.01)
    v.append(2)
    v.append(2.5)
    v.append(3)
    return np.diag(v)
def construct_T(alpha, beta):
    if beta is None:
        return alpha[0]
    return np.diag(alpha) + np.diag(beta, k=1) + np.diag(beta, k=-1)
def lanczos(A, N):
    m = len(A)
    b = np.random.rand(m, 1)
    q_n = b/np.linalg.norm(b, 2)
    beta_n_ = 0
    q_n_ = np.zeros((m, 1))
    alpha, beta = [], []
    eigs = np.zeros((N, N))
    for n in range(N):
        z = np.zeros((1, N))
        v = A @ q_n
        alpha_n = (q_n.T @ v).item()
        v = v - alpha_n * q_n - beta_n_ * q_n_
        beta_n = np.linalg.norm(v, 2)
        # Construct tridiagonal matrix and get eigenvalues
        alpha.append(alpha_n)
        T = construct_T(alpha, beta)
        eig, vec = np.linalg.eig(T)
        eig = np.array(sorted(eig))
        z[0, :eig.shape[0]] = eig.reshape(1, len(eig))
        eigs[n, :] = z
        # Update for next loop
        q_n_ = q_n
        q_n = v / beta_n
        beta_n_ = beta_n
        beta.append(beta_n)
    return alpha, beta
def lanczos_orth(A, x, N):
    m = len(A)
    q = x/np.linalg.norm(x, 2)
    Q_n = copy.deepcopy(q)
    r = A @ q
    alpha_n = (q.T @ r).item()
    r = r - alpha_n * q
    beta_n = np.linalg.norm(r, 2)
    eigs = np.zeros((N, N))
    for n in range(1, N):
        z = np.zeros((1, N))
        v = q
        q = r / beta_n
        Q_n = np.column_stack((Q_n, q))
        r = A @ q - beta_n * v
        alpha_n = (q.T @ r).item()
        # Reorthogonalize
        r = r - Q_n @ (Q_n.T @ r)
        r = r - Q_n @ (Q_n.T @ r)
        beta_n = np.linalg.norm(r, 2)
        # Get eigenvalues
        T = Q_n.T @ A @ Q_n
        eig, vec = np.linalg.eig(T)
        eig = np.array(sorted(eig))
        z[0, :eig.shape[0]] = eig.reshape(1, len(eig))
        eigs[n, :] = z
    return T, Q_n
def plot_triangular_mat(mat, scale, color_cyc, ylabel, title):
    N = len(mat)
    wid = -.02 * N + 3.4
    ax = plt.figure().gca()
    ax.set_prop_cycle(cycler('color', color_cyc) + cycler('lw', [wid]*len(color_cyc)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    flag = -1 # positive means upper, negative means lower
    for i in range(N):
        if i == 0:
            eig_list = mat[0, :].tolist()
        else:
            if flag == -1:
                mat = mat[:-1, 1:]
                eig_list = [j for j in mat[0, :].tolist() if j != 0]
            else:
                mat = mat[1:, 1:]
                eig_list = np.diag(mat).tolist()
        flag *= -1
        idx_list = [k-1 for k in range(N, N - len(eig_list), -1)][::-1]
        plt.plot(idx_list, eig_list)
        plt.plot(idx_list, eig_list, '*', markersize = wid)
    plt.yscale(scale)
    plt.xlabel('Step Number $n$')
    plt.ylabel(ylabel)
    plt.show()
    # plt.savefig(title)
def plot_err(eigs, eig_true, scale):
    N = len(eigs)
    plt.yscale("log")
    err_mat = np.zeros((N, N))
    for i in range(N):
        eigs_k = eigs[:, i]
        eigs_k = [j for j in eigs_k.tolist() if j != 0]
        for j, eig in enumerate(eigs_k):
            err = abs(eig - eig_true[min(range(len(eig_true)), key = lambda i: abs(eig_true[i]-eig))])
            err_mat[i, j] = err
    # plot_triangular_mat(err_mat.T, scale, ['y', 'k', 'y', 'k'],  'Error','Lanczos_err_orth_'+str(scale)+str(N)+'.png' )
    err_tot, ks = [], []
    for k in range(N):
        err_tot.append(sum(err_mat[:, k]))
        ks.append(k)
    return ks, err_tot
# if __name__ == '__main__':
#     N = 200
#     A = lanc_matrix()
#     w, v = np.linalg.eig(A)
#     eigs_orth = lanczos_orth(A, N)
#     eigs = lanczos(A, N)
#     # plot_triangular_mat(eigs, "linear", ['y', 'k', 'y', 'k'], 'Ritz Values', 'Lanczos_orth'+str(N)+'.png')
#     ks1, err1 = plot_err(eigs_orth, w, 'log')
#     ks2, err2 = plot_err(eigs, w, 'log')
#     plt.plot(ks1, err1, 'b', label='Lanczos Orthogonal')
#     plt.plot(ks2, err2, 'm', label='Lanczos')
#     plt.xlabel('Step Number $n$')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.savefig('lanc_comp.png')
