import numpy as np
import matplotlib.pyplot as plt

def data(n, p):
    n = 1000
    p = 10
    x = np.random.random((p, n))
    y = np.zeros((p, p))
    y[:, 0] = x[:, 0]
    mu = np.zeros(p)
    cov = np.eye(p)
    xi = np.random.multivariate_normal(mu, cov, size=p)
    z = y + xi
    return x, z
def prob5():
    n = 1000
    x, z = data(n, p)

if __name__ == '__main__':
    prob5()
