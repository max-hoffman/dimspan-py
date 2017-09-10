from functools import reduce
from math import factorial
import numpy as np

def constrainedSparsify(constraints, polyorder, modes, theta, dX, _lambda, dimensions):
  """
  Sparsify Xi, but with special constraints for known variables.abs

  Constraints : array of tuples representing the functions to fix
    - ex: [['a','b'],['ab']] -> fix the functions dx/dt = x + y, and dy/dt = xy
      for some parameters 
  """

  matches = _constraintMatches(len(constraints), polyorder)
  print("matches", matches)

  constrainedXi = _marshalXi(constraints, polyorder, modes)
  print("constrained xi", constrainedXi)

  xi, residual, _, _ = np.linalg.lstsq(theta, dX)
  lowestResidual = np.sum(residual)
  bestXi = np.copy(xi)

  for colsToFix in matches:
    xi, _, _, _ = np.linalg.lstsq(theta, dX)

    # TODO : make sure this isn't jenky
    # xi[:, colsToFix] = constrainedXi
    for i in range(len(colsToFix)):
      xi[:, colsToFix[i]] = constrainedXi[:, i]
    
    for i in range(10):
      xi = _regularize(xi, _lambda)
      xi, residual = _iterateRegression(xi, theta, dX, _lambda, dimensions)
    
    currentResidual = np.sum(residual)
    if currentResidual < lowestResidual:
        lowestResidual = currentResidual
        bestXi = np.copy(xi)

    print("constrained xi after optimization", xi)
    print("guess, residual", colsToFix, currentResidual)
  return bestXi, lowestResidual

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
      tempXi, residuals, _, _ = np.linalg.lstsq(theta[:, bigIdx], dX[:, dim])
      xi[bigIdx, dim] = tempXi
  
  return xi, residuals

def _sumOfComboWithReplacement(total, options):
  "Helper function that gives reference for polynomial functions counts"
  count = 1
  for i in range(1, total+1):
    count += factorial(options+i-1) / (factorial(i) * factorial(options-1))
  return int(count)
  
def _stringSum(string):
  "Helper function to condense function string to numerical representation"
  chars = list(string)
  ints = list(map(lambda x: (int(format(ord(x), "x"))-61), chars))
  return reduce(
    (lambda accum, curr: accum + curr), ints, 0
  )

def _indexForString(string, modes):
  "Returns the Xi index for a given string"
  order = len(string)
  ref = _sumOfComboWithReplacement(order-1, modes)
  return ref + _stringSum(string)

def _marshalXi(constraints, order, modes):
  "Converts string representation to sparse representation"
  rowCount = len(constraints)
  colCount = _sumOfComboWithReplacement(order, modes)
  xi = np.zeros((rowCount,colCount))

  for row in range(rowCount):
    for varString in constraints[row]:
      idx = _indexForString(varString, modes)
      xi[row, idx] = 1
  
  return xi.T

def _constraintMatches(fixed, order):
  """
  Given the number of functions we want to sparsily fix, and the total
  possible number of dimensions we are optimizing over, return an array holding
  tuples of all of the possible sets of indices for which we can fix
  for that optimization.
  """
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
      
      innerRecurse(newCopy, newRemaining)

  def duplicate(arr):
    copy = list(arr)
    copy.sort()
    for idx in range(1, len(copy)):
      if copy[idx] == copy[idx - 1]:
        return False
    return True

  if fixed > order:
    return

  remaining = list(range(order))
  current = []
  matches = []

  innerRecurse([], remaining)
    
  return list(filter((lambda arr: duplicate(arr)), matches))
