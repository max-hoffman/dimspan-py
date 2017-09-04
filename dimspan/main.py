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
print "original data shape", lorenzData.shape
plt.plot(lorenzData[:, 0], lorenzData[:, 1], 'r-')
pylab.show()

# reconstruct lorenz data from single time series
henkeledLorenz = henkelify(lorenzData[:, 0], 10)
U, s, V = svd(henkeledLorenz, full_matrices=False)
print "U shape", U.shape
print s
plt.plot(U[:,0], U[:, 1], 'r-')
pylab.show()

# make and normalize theta
theta = poolData(U, 3, 3, False)
print "pre-normalized theta", theta
theta, norms = normalize(theta, 3)
print "norms", norms
theta = theta[2:len(theta)-2, :]
print "theta shape: ", theta.shape
print "normalized theta", theta
raw_input("Press any key to : compute derivatives...")

# compute derivatives
dV = fourthOrderDerivative(U, .001, 3)
print "dV shape: ", dV.shape
print "dV", dV
raw_input("Press any key to : test least-squares step...")

# solve theta * xi - dV = 0 with partial least-squares
xi, resid, rank, s = np.linalg.lstsq(theta, dV)

print "xi", xi
print "resid", resid
print "rank", rank
print "s", s
raw_input("Press any key to : sparsify xi...")

xi = sparsify(theta, dV, 100, 3)
print "optimize xi", xi