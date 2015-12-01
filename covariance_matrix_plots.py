import numpy as np
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel
import matplotlib.pyplot as plt

# make up some xs
x = np.arange(50)
yerr = np.ones_like(x) * .1

# set up a SE GP
A, l = 100, 10
K = A**2 * ExpSquaredKernel(l**2)
gp = george.GP(K)
gp.compute(x, yerr)
k = gp.get_matrix(x)

# And plot it...
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(k, cmap=plt.cm.gray)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig("SEmatrix")

# set up a QP GP
A, l, gamma, P = 100, 10, 1, 10
K = A**2 * ExpSquaredKernel(l**2) * ExpSine2Kernel(gamma, P)
gp = george.GP(K)
gp.compute(x, yerr)
k = gp.get_matrix(x)

# And plot it...
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(k, cmap=plt.cm.gray)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig("QPmatrix")
