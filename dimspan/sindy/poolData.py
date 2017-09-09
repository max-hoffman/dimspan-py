import numpy as np
from math import factorial

def poolData(inputData, variableCount, polyorder, usesine):
  """creates "theta" matrix of all possible linear combinations
  of input array up to the polynomial order indicated
  """

  rowCount = inputData.shape[0]
  colCount = sumOfComboWithReplacement(polyorder, variableCount)
  if usesine:
    harmonicCount = 10
    colCount += harmonicCount * 2 * variableCount 

  theta = np.zeros((int(rowCount),int(colCount)))

  colIdx = 0
  theta[:,colIdx] = np.ones(theta.shape[0])
  colIdx += 1
  
  for i in range(variableCount):
    theta[:, colIdx] = inputData[:, i]
    colIdx += 1

  if polyorder>=2:
    for i in range(variableCount):
      for j in range(i, variableCount):
        theta[:,colIdx] = recursiveMultiply(inputData, [i, j])
        colIdx += 1

  if polyorder>=3:
    for i in range(variableCount):
        for j in range(i, variableCount):
          for k in range(j, variableCount):
            theta[:,colIdx] = recursiveMultiply(inputData, [i, j, k])
            colIdx += 1

  if polyorder>=4:
    for i in range(variableCount):
      for j in range(i, variableCount):
        for k in range(j, variableCount):
          for l in range(k, variableCount):
            val = recursiveMultiply(inputData, [i, j, k, l])
            theta[:,colIdx] = val
            colIdx += 1

  if polyorder>=5:
    for i in range(variableCount):
      for j in range(i, variableCount):
        for k in range(j, variableCount):
          for l in range(k, variableCount):
            for m in range(l, variableCount):
              theta[:,colIdx] = recursiveMultiply(inputData, [i, j, k, l, m])
              colIdx += 1

  if usesine:
    for k in range(1, 11):
      sinedVals = np.sin(inputData[:, :variableCount] * k)
      theta[:, colIdx:colIdx+variableCount] = sinedVals
      colIdx += variableCount

      cosedVals = np.cos(inputData[:, :variableCount] * k)
      theta[:, colIdx:colIdx+variableCount] = cosedVals
      colIdx += variableCount
  return theta

def recursiveMultiply(inputData, indices):
  def innerRecurse(inputVectors, accumulated):
    if len(inputVectors) > 0:
      return innerRecurse(inputVectors[1:], np.multiply(accumulated, inputVectors[0]))
    return accumulated

  vectors = np.zeros((len(indices), inputData.shape[0]))

  for i in range(len(indices)):
    vectors[i] = inputData[:, indices[i]]

  startVec = np.ones(inputData[:, i].shape[0])

  return innerRecurse(vectors, startVec)

def sumOfComboWithReplacement(total, options):
  count = 1
  for i in range(1, total+1):
    count += factorial(options+i-1) / (factorial(i) * factorial(options-1))
  return count