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

  matches = _constraintMatches(len(constraints), 4)
  print("matches", matches)

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

  remaining = list(range(order+1))
  current = []
  matches = []

  def innerRecurse(current, remaining):
    
    if current:
      if len(current) == fixed:
        matches.append(current)
        return
      
    for idx in range(len(remaining)):
      newCopy = list(current)
      newCopy.append(remaining[idx])
      newRemaining = list(remaining[0:idx])  + list(remaining[idx+1:])
      
      if not current:
        newCopy = [remaining[idx]]
      
      print("recurse", newCopy, newRemaining)
      innerRecurse(newCopy, newRemaining)

  innerRecurse([], remaining)

  def duplicate(arr):
    copy = list(arr)
    copy.sort()
    for idx in range(1, len(copy)):
      if copy[idx] == copy[idx - 1]:
        return False
    return True
    
  return list(filter((lambda arr: duplicate(arr)), matches))



