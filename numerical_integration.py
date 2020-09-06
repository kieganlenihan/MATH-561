from math import exp, log
import time
## Inputs
f = lambda x: exp(-x**2/2)
a = -1
b = 2.5
n = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
## Functions
def num_int(a, b, n):
    dx = (b-a)/n
    x_l, left_sum, mid_sum, trap_sum = a, 0, 0, 0
    mid_start = time.time()
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
def int_itr(a, b, int_pts):
    left_int, mid_int, trap_int = [], [], []
    left_times, mid_times, trap_times = [], [], []
    for n in int_pts:
        left, mid, trap, times = num_int(a, b, n)
        left_times.append(times[0])
        mid_times.append(times[1])
        trap_times.append(times[2])
        left_int.append(left)
        mid_int.append(mid)
        trap_int.append(trap)
    return [left_int, mid_int, trap_int], [left_times, mid_times, trap_times]
# Get integrals
sums, times = int_itr(a, b, n)
print(times)
# Get accurate integral
from sympy import *
import inspect
init_printing(use_unicode=False, wrap_line=False)
funcStr = str(inspect.getsourcelines(f)[0])
func = funcStr.strip("['\\n']").split(": ")[1]
x = Symbol('x')
acc = integrate(func, (x, a, b)).evalf()
# Plot
import numpy as np
import matplotlib.pyplot as plt
plt.xscale('log')
plt.yscale('log')
colors = ['b', 'r', 'g']
leg = ['left-point Riemann sum', 'mid-point Riemann sum', 'trapezoidal rule']
conv = []
for i in range(3):
    int = times[i]
    # err = []
    # for j in range(len(n)):
    #     err.append(abs(acc-int[j]))
    plt.scatter(n, int, color=colors[i])
    n_log = np.asarray([log(x) for x in n]).astype(np.float32)
    err_log = np.asarray([log(x) for x in int]).astype(np.float32)
    conv.append(np.polyfit(n_log, err_log, 1)[0])
plt.legend(leg)
plt.grid()
plt.xlabel('n')
plt.ylabel('Time')
# plt.title('Error vs n')
plt.savefig('num_int_time.png')
plt.show()
