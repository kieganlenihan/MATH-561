from math import exp, log
import numpy as np
import time
## Inputs
a = -1
b = 2.5
n = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
## Functions
def f(x):
    return np.exp(-x**2/2)
def left_int(a, b, n):
    dx = (b - a)/n
    x = np.arange(a, b, dx)
    return f(x).sum()*dx
def mid_int(a, b, n):
    dx = (b - a)/n
    x1 = np.arange(a, b, dx)
    x2 = np.arange(a+dx, b+dx, dx)
    return f(x1+x2).sum()*dx
def trap_int(a, b, n):
    dx = (b - a)/n
    x = np.arange(a,b,dx)
    return f(x).sum()*dx - f(a)/2*dx - f(b)/2*dx
def num_int(a, b, n):
    mid_start = time.time()
    dx = (b-a)/n
    x_l, left_sum, mid_sum, trap_sum = a, 0, 0, 0
    for i in range(n):
        x_r = x_l + dx
        mid_sum += f((x_l+x_r)/2)*dx
        x_l = x_r
    mid_t = time.time()-mid_start
    left_trap_start = time.time()
    x_l = a
    for i in range(n):
        left_sum += f(x_l)*dx
        x_l += dx
    left_t = time.time()-left_trap_start
    trap_sum = left_sum-(f(a)-f(b))/2*dx
    trap_t = time.time()-left_trap_start
    return left_sum, mid_sum, trap_sum, [left_t, mid_t, trap_t]
def timer(a, b, pts):
    vec_times, num_times = [], []
    for n in pts:
        start = time.time()
        left_int(a, b, n)
        left_t = time.time()-start
        start = time.time()
        mid_int(a, b, n)
        mid_t = time.time()-start
        start = time.time()
        trap_int(a, b, n)
        trap_t = time.time()-start
        start = time.time()
        _, _, _, times = num_int(a, b, n)
        vec_times.append([left_t, mid_t, trap_t])
        num_times.append(times)
    return vec_times, num_times
vt, ft = timer(a, b, n)
ft = list(map(list, zip(*ft)))
vt = list(map(list, zip(*vt)))
## Plot
import matplotlib.pyplot as plt
plt.xscale('log')
plt.yscale('log')
colors = ['b', 'r', 'g']
leg = ['Vector sum', 'For loop sum']
conv = []
for i in range(3):
    v = vt[i]
    f = ft[i]
    plt.scatter(n, v, color='b')
    plt.scatter(n, f, color='r')
plt.legend(leg)
plt.grid()
plt.xlabel('n')
plt.ylabel('Time')
plt.savefig('vec_vs_for_time.png')
plt.show()
