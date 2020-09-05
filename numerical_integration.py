from math import exp, log
import time
## Inputs
f = lambda x: exp(-x**2/2)
a = -1
b = 2.5
n = [8, 16, 32, 64, 128, 256, 512, 1024]
## Functions
def num_int(a, b, n):
    dx = (b-a)/n
    x_l, left_sum, mid_sum, trap_sum = a, 0, 0, 0
    for i in range(n):
        x_r = x_l + dx
        left_sum += f(x_l)*dx
        mid_sum += f((x_l+x_r)/2)*dx
        x_l = x_r
    trap_sum = left_sum-(f(a)-f(b))/2*dx
    return left_sum, mid_sum, trap_sum
def int_itr(a, b, int_pts):
    left_int, mid_int, trap_int, time_list = [], [], [], []
    for n in int_pts:
        start = time.time()
        left, mid, trap = num_int(a, b, n)
        time_list.append(time.time()-start)
        left_int.append(left)
        mid_int.append(mid)
        trap_int.append(trap)
    return [left_int, mid_int, trap_int], time_list
# Get integrals
start_t = time.time()
sums, time_list = int_itr(a, b, n)
total_t = time.time()-start_t
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
    int = sums[i]
    err = []
    for j in range(len(n)):
        err.append(abs(acc-int[j]))
    plt.scatter(n, err, color=colors[i])
    n_log = np.asarray([log(x) for x in n]).astype(np.float32)
    err_log = np.asarray([log(x) for x in err]).astype(np.float32)
    conv.append(np.polyfit(n_log, err_log, 1)[0])
plt.legend(leg)
plt.grid()
plt.xlabel('n')
plt.ylabel('Error')
plt.title('Error vs n')
plt.savefig('num_int.png')
plt.show()
