import numpy as np
from scipy import integrate

def lorenz(sigma, beta, rho, timeStart, timeStop, numberOfPoints, dimensions):
  "returns lorenz data at the timescale, timespan and dimension indicated"
  
  def lorenzeODE(X, t=0):
    xDot = sigma * (X[1] - X[0])
    yDot = X[0] * (rho - X[2]) - X[1]
    zDot = X[0] * X[1] - beta * X[2]
    return [xDot, yDot, zDot]

#  time = np.linspace(timeStart, timeStop, num = numberOfPoints)
  time = np.arange(0, timeStop, timeStart)
  """ this makes it literally the same as the matlab implementation, dt=timestart JKJ"""
  numberOfPoints=len(time)
  initialCond = np.array([-8, 8, 27])

  lorenzVals, infodict = integrate.odeint(lorenzeODE, initialCond, time, full_output=1)
  print( infodict['message'])

  return lorenzVals[:, :dimensions]

def noisify(numpyMatrix, magnitude):
  "returns array with normalized noise added"

  for val in np.nditer(numpyMatrix, op_flags=['readwrite']):
    val += np.random.rand()

def fourthOrderDerivative(inputData, dt, dimensions):
  numDer = np.zeros((inputData.shape[0], dimensions))

  for c in range(dimensions):
    for r in range(numDer.shape[0]):
      if r < 2 or r >= numDer.shape[0] - 2:
        continue

      currentAppx = (1 / (12 * dt)) * (-inputData[r+2][c] + 8*inputData[r+1][c] - 8*inputData[r-1][c] + inputData[r-2][c])
      numDer[r, c] = currentAppx

  return numDer[2:len(numDer)-2, :]

def henkelify(numpyArray, rowCount):
  "morphs single dimension input array into a matrix with delta ROWS"

  length = numpyArray.shape[0]
  colCount = length - rowCount
  henkeled = np.zeros((rowCount, colCount))

  for row in range(rowCount):
    start = row
    end = colCount + row
    henkeled[ row,:] = numpyArray[start:end]
  
  return henkeled

def normalize(inputData, dimensions):
  norms = np.zeros(dimensions)
  normalizedData = np.copy(inputData)

  for dim in range(dimensions):
    newNorm = np.linalg.norm(inputData[:,dim], ord=2) # / len(inputData)
    normalizedData[:, dim] = normalizedData[:, dim] / newNorm
    norms[dim] = newNorm

  return normalizedData, norms