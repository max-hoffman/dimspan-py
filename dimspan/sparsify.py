import numpy as np

def sparsify(theta, dX, _lambda, dimensions):
  xi, _, _, _ = np.linalg.lstsq(theta, dX)
  
  for i in range(10):
    xi = _regularize(xi, _lambda)
    xi = _iterateRegression(xi, theta, dX, _lambda, dimensions)
  return xi

def _regularize(xi, _lambda):
  if isinstance(xi[0], float):
      smallIdx = [i for i,v in enumerate(xi) if abs(v) <= _lambda]
      xi[smallIdx] = 0
  else: 
      for dim in range(dimensions):
        smallIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) <= _lambda]
        xi[smallIdx, dim] = 0
  return xi

def _iterateRegression(xi, theta, dX, _lambda, dimensions):
  if dimensions == 1:
    bigIdx = [i for i,v in enumerate(xi) if abs(v) > _lambda]
    tempXi, _, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX)
    xi[bigIdx] = tempXi

  else:
    for dim in range(dimensions):
      bigIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) > _lambda]
      tempXi, _, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX[:, dim])
      xi[bigIdx, dim] = tempXi
  return xi