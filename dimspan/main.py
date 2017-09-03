"""Python implementation of SINDy algorithm"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pylab

from helpers import *

# test
lorenzData = lorenz(10, 8/3, 28, .001, 1000, 100000, 3)
noisify(lorenzData, 5)

plt.plot(lorenzData[:, 0], lorenzData[:, 1], 'r-')
pylab.show()

# make henkel matrix
def henkelify(numpyArray, delta):
  "morphs single dimension input array into a matrix with delta columns"

henkeledLorenz = henkelify(lorenzData[:,0])

# perform SVD on that matrix
U, s, V = np.linalg.svd(henkeledLorenz, full_matrices=True)

# pool data function
def poolData(numpyArray, columnsToPool, polyorder, usesine):
  """creates "theta" matrix of all possible linear combinations
  of input array up to the polynomial order indicated
  """

# get norm function - already exists for numpy

# solve theta * xi - dV = 0 with partial least-squares
  # divide xi columns by the norms from theta cols