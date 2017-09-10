"""Python implementation of SINDy algorithm"""

import numpy as np
from scipy import integrate
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pylab

import sindy

# raw data
lorenzData = sindy.lorenz(10, 8/3, 28, .001, 100, 10000, 3)
#noisify(lorenzData, 1)
print( "original data shape", lorenzData.shape)
# plt.plot(lorenzData[:, 0], lorenzData[:, 1], 'r-')
print( "Exit plot window to continue...")
# pylab.show()

# reconstruct lorenz data from single time series
henkeledLorenz = sindy.henkelify(lorenzData[:, 0], 10)
""" the henkel was transposed from what it needs to be JKJ"""
U, s, V = svd(henkeledLorenz, full_matrices=False)
""" it looks like scipy's linalg gives the true SVD where as matlab babies the user by taking the conjugate transpose for them JKJ"""
V=V.conj().T 
print( "V shape", V.shape)
print( s)
# plt.plot(V[:,0], V[:, 1], 'r-')
print( "Exit plot window to continue...")
# pylab.show()
#raw_input("Press any key to : compute derivatives...")

# compute derivatives
numModes=3 
""" they cheat and keep only 3 modes thereby forcing the others to be zero (it works without cheating too) JKJ """
dV = sindy.fourthOrderDerivative(V, .001, numModes)
print( "dV shape: ", dV.shape)
print( "dV", dV)

# make and normalize theta
""" dV and V have to be the same size so they change that in the matlab implementation JKJ """
theta = sindy.poolData(V[2:-2,0:numModes], numModes, 3 , False)
print( "pre-normalized theta", theta[0, :])

theta, norms = sindy.normalize(theta, theta.shape[1])
""" the normalization was not correct it had an extra division by length JKJ"""
print( "norms", norms)
#theta = theta[2:len(theta)-2, :]
print( "theta shape: ", theta.shape)
print( "normalized theta", theta[0, :])
#raw_input("Press any key to : test least-squares step...")

# solve theta * xi - dV = 0 with partial least-squares
xi, resid, rank, s = np.linalg.lstsq(theta, dV)

print( "xi", xi)
print( "resid", resid)
print( "rank", rank)
print( "s", s)
# raw_input("Press any key to : sparsify xi...")

xi[:, 0] = sindy.sparsify(theta, dV[:, 0], .01, 1)
xi[:, 1] = sindy.sparsify(theta, dV[:, 1], .2, 1)
xi[:, 2] = sindy.sparsify(theta, dV[:, 2], 2, 1)

""" now you need to renormalize sparsified junk with the norms from early JKJ """
xi[:, 0] = xi[:, 0] / norms[0]
xi[:, 1] = xi[:, 1] / norms[1]
xi[:, 2] = xi[:, 2] / norms[2]

print( "optimize xi", xi)

constraints = [['a', 'b'], ['a', 'b', 'ac']]

polyorder = 3
xi, residual = sindy.constrainedSparsify(constraints, polyorder, numModes, theta, dV, .01, 3)
print("contstrained xi with lowest residual", xi)
print("lowest constrained residual", residual)