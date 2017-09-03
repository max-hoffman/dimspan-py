import numpy as np

def poolData(inputArray, columnsToPool, polyorder, usesine):
  """creates "theta" matrix of all possible linear combinations
  of input array up to the polynomial order indicated
  """

  rowCount = inputArray.shape[0]
  colCount = 1+columnsToPool+(columnsToPool*(columnsToPool+1)/2)+(columnsToPool*(columnsToPool+1)*(columnsToPool+2)/(2*3))+11
  theta = np.zeros((rowCount,colCount))

  colIdx = 0
  # % poly order 0
  theta[:,colIdx] = np.ones(theta.shape[0])
  colIdx += 1
  
  # % poly order 1
  for i in range(columnsToPool):
    theta[:,colIdx] = inputArray[:, i]
    colIdx += 1


  if polyorder>=2:
    for i in range(columnsToPool):
      for j in range(i, columnsToPool):
        theta[:,colIdx] = recursiveMultiply(inputArray, [i, j])
        colIdx += 1

  if polyorder>=3:
    for i in range(columnsToPool):
        for j in range(i, columnsToPool):
          for k in range(j, columnsToPool):
            currentVecs = np.array([inputArray[:, i], inputArray[:, j], inputArray[:, k]])
            theta[:,colIdx] = recursiveMultiply(inputArray, [i, j, k])
            colIdx += 1

  if polyorder>=4:
    for i in range(columnsToPool):
      for j in range(i, columnsToPool):
        for k in range(j, columnsToPool):
          for l in range(k, columnsToPool):
            currentVecs = np.array([inputArray[:, i], inputArray[:, j], inputArray[:, k], inputArray[:, l]])
            theta[:,colIdx] = recursiveMultiply(inputArray, [i, j, k, l])
            colIdx += 1

  if polyorder>=5:
    for i in range(columnsToPool):
      for j in range(i, columnsToPool):
        for k in range(j, columnsToPool):
          for l in range(k, columnsToPool):
            for m in range(l, columnsToPool):
              currentVecs = np.array([inputArray[:, i], inputArray[:, j], inputArray[:, k], inputArray[:, l], inputArray[:, m]])
              theta[:,colIdx] = recursiveMultiply(inputArray, [i, j, k, l, m])
              colIdx = colIdx+1

  return theta

# if(usesine)
#     for k=1:10;
#         yout = [yout sin(k*inputArray) cos(k*inputArray)];
#     end
# end

def recursiveMultiply(inputArray, indices):
  def innerRecurse(inputVectors, accumulated):
    if len(inputVectors) > 0:
      return innerRecurse(inputVectors[1:], np.multiply(accumulated, inputVectors[0]))
    return accumulated

  vectors = np.zeros((len(indices), inputArray.shape[0]))

  for i in range(len(indices)):
    vectors[i] = inputArray[:, indices[i]]

  startVec = np.ones(inputArray[:, i].shape[0])

  return innerRecurse(vectors, startVec)