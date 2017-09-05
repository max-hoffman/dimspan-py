import numpy as np

def sparsify(theta, dX, _lambda, dimensions):
  """sparse solution for dX/dt = theta(X) * Xi

    @param theta : each col is polynomially expanded function of row data
    @param dX : approximate derivative of time-series data

    returns xi : sparse approximation for dynamics underlying time series data
  """
  xi, _, _, _ = np.linalg.lstsq(theta, dX)
  
  for i in range(10):
    xi = _regularize(xi, _lambda)
    xi = _iterateRegression(xi, theta, dX, _lambda, dimensions)
  return xi

def _regularize(xi, _lambda):
  "clips small elements (to zero)"

  if isinstance(xi[0], float):
      smallIdx = [i for i,v in enumerate(xi) if abs(v) <= _lambda]
      xi[smallIdx] = 0
  else: 
      for dim in range(dimensions):
        smallIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) <= _lambda]
        xi[smallIdx, dim] = 0
  return xi

def _iterateRegression(xi, theta, dX, _lambda, dimensions):
  "performs least-squares iteration on a subset of indices above lambda threshold"

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