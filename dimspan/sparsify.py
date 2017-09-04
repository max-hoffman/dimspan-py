import numpy as np

def sparsify(theta, dX, _lambda, dimensions):
  xi, _, _, _ = np.linalg.lstsq(theta, dX)

  for i in range(10):
    for dim in range(dimensions):
      smallIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) <= _lambda]
      xi[smallIdx, :] = 0

    for dim in range(dimensions):
      bigIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) > _lambda]
      tempXi, _, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX[:, dim])
      xi[bigIdx, dim] = tempXi
  return xi
