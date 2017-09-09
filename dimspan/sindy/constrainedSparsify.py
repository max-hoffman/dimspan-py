from functools import reduce
from math import factorial
import numpy as np

def _sumOfComboWithReplacement(total, options):
  count = 1
  for i in range(1, total+1):
    count += factorial(options+i-1) / (factorial(i) * factorial(options-1))
  return int(count)
  
def _stringSum(string):
  chars = list(string)
  ints = list(map(lambda x: (int(format(ord(x), "x"))-61), chars))
  return reduce(
    (lambda accum, curr: accum + curr), ints, 0
  )

def _indexForString(string, modes):
  order = len(string)
  ref = _sumOfComboWithReplacement(order-1, modes)
  return ref + _stringSum(string)

def _marshalXi(constraints, order, modes):
  rowCount = len(constraints)
  colCount = _sumOfComboWithReplacement(order, modes)
  xi = np.zeros((rowCount,colCount))

  for row in range(rowCount):
    for varString in constraints[row]:
      idx = _indexForString(varString, modes)
      xi[row, idx] = 1
  
  return xi.T

def constrainedSparsify(constraints, polyorder, modes, theta, dX, _lambda, dimensions):
  for 
  constrainedXi = _marshalXi(constraints, polyorder, modes)
  xi, _, _, _ = np.linalg.lstsq(theta, dX)
  
  for i in range(10):
    xi = _regularize(xi, _lambda)
    xi = _iterateRegression(xi, theta, dX, _lambda, dimensions)
  return xi


def _regularize(xi, _lambda):
  "clips small elements (to zero)"

  smallIdx = abs(xi)<=_lambda
  xi[smallIdx] = 0
  return xi

def _iterateRegression(xi, theta, dX, _lambda, dimensions):
  "performs least-squares iteration on a subset of indices above lambda threshold"

  if dimensions == 1:
    bigIdx = abs(xi[:]) > _lambda
    tempXi, _, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX)
    xi[bigIdx] = tempXi
    
  else:
    for dim in range(dimensions):
      # bigIdx = [i for i,v in enumerate(xi[:, dim]) if abs(v) > _lambda]
      bigIdx = abs(xi[:, dim]) > _lambda
      tempXi, _, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX[:, dim])
      xi[bigIdx, dim] = tempXi
  
  return xi

def _constraintMatches(fixed, order):

  if fixed > order:
    return

  remaining = list(range(fixed))
  current = []
  matches = []

  def innerRecurse(current, remaining):
    if len(current) == order:
      matches.push(current)
      return
    for idx in remaining:
      innerRecurse(current.push(remaining[0]), remianing[1:])

  innerRecurse(current, remaining)
  return matches




