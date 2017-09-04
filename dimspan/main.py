"""Python implementation of SINDy algorithm"""

import numpy as np
from scipy import integrate
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pylab

from helpers import *
from poolData import poolData
from sparsify import sparsify

# raw data
lorenzData = lorenz(10, 8/3, 28, .001, 100, 10000, 3)
noisify(lorenzData, 1)
# plt.plot(lorenzData[:, 0], lorenzData[:, 1], 'r-')
# pylab.show()

henkeledLorenz = henkelify(lorenzData[:, 0], 2000)
U, s, V = svd(henkeledLorenz, full_matrices=False)
# print s
# plt.plot(U[:,0], U[:, 1], 'r-')
# pylab.show()

# make and normalize theta
theta = poolData(V, 3, 3, False)
print "pre-normalized theta", theta
theta, norms = normalize(theta, 3)
print norms
theta = theta[2:len(theta)-2, :]
print "normalized theta", theta

# compute derivatives
dV = fourthOrderDerivative(V, .001, 3)
dX = dV[2:len(dV)-2, :]
print "dX", dX[0:5, :]

# solve theta * xi - dV = 0 with partial least-squares
xi, resid, rank, s = np.linalg.lstsq(theta, dX)

print "xi", xi
print "resid", resid
print "rank", rank
print "s", s

xi = sparsify(theta, dX, 100, 3)
print "optimize xi", xi