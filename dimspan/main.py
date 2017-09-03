"""Python implementation of SINDy algorithm"""

import numpy as np
from scipy import integrate
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pylab

from helpers import *
from sindyFile import *


# test
lorenzData = lorenz(10, 8/3, 28, .001, 100, 10000, 3)
noisify(lorenzData, 5)

plt.plot(lorenzData[:, 0], lorenzData[:, 1], 'r-')
# pylab.show()

henkeledLorenz = henkelify(lorenzData[:, 0], 2000)

# perform SVD on that matrix
# U, s, V = svd(henkeledLorenz, full_matrices=False)
# print s
# plt.plot(U[:,0], U[:, 1], 'r-')
# pylab.show()

theta = poolData(henkeledLorenz, 3, 4, False)
print theta[0, :30]
print theta.shape

# get norm function - already exists for numpy

# solve theta * xi - dV = 0 with partial least-squares
  # divide xi columns by the norms from theta cols