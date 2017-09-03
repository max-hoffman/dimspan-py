import numpy as np
from scipy import integrate

# create lorenz data
def lorenz(sigma, beta, rho, timeStart, timeStop, numberOfPoints, dimensions):
  "returns lorenz data at the timescale, timespan and dimension indicated"
  
  def lorenzeODE(X, t=0):
    xDot = sigma * (X[1] - X[0])
    yDot = X[0] * (rho - X[2]) - X[1]
    zDot = X[0] * X[1] - beta * X[2]
    return [xDot, yDot, zDot]

  time = np.linspace(timeStart, timeStop, num = numberOfPoints)
  initialCond = np.array([-8, 8, 27])

  lorenzVals, infodict = integrate.odeint(lorenzeODE, initialCond, time, full_output=1)
  print infodict['message']
  print lorenzVals

  return lorenzVals[:, :dimensions]

# add noise function
def noisify(numpyMatrix, magnitude):
  "returns array with normalized noise added"

  for val in np.nditer(numpyMatrix, op_flags=['readwrite']):
    val += np.random.rand()

# 4th order derivative function
def fourthOrderDerivative(numpyMatrix, dt, dimensions):
  numDer = np.copy(numpyMatrix)

  for r, c = range(numDer.shape):
    if r < 2 or r >= numDer.shape[0] - 2:
      continue

    if c >= dimensions:
      return numDer

    currentAppx = (1 / (12 * dt)) * (-numDer[r+2][c] + 8*numDer[r+1][c] - 8 - numDer[r-1][c] + numDer[r-2][c])
    numDer[i][j] = currentAppx

  return numDer

# make henkel matrix
def henkelify(numpyArray, delta):
  "morphs single dimension input array into a matrix with delta columns"
