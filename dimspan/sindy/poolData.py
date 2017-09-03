import numpy as np

def poolData(inputArray, columnsToPool, polyorder, usesine):
  """creates "theta" matrix of all possible linear combinations
  of input array up to the polynomial order indicated
  """

  rowCount = inputArray.shape[0]
  theta = np.zeros(rowCount,1+columnsToPool+(columnsToPool*(columnsToPool+1)/2)+(columnsToPool*(columnsToPool+1)*(columnsToPool+2)/(2*3))+11)

  colIdx = 0
  # % poly order 0
  theta[:,colIdx] = np.ones((theta.shape[0], 1))
  colIdx += 1
  
  # % poly order 1
  for i in range(columnsToPool):
      theta[:,colIdx] = inputArray[:,i]
      colIdx += 1


  if polyorder>=2:
      for i in range(columnsToPool):
          for j in range(i, columnsToPool):
              theta[:,colIdx] = np.multiple(inputArray[:,i], inputArray[:,j])
              colIdx += 1

# if(polyorder>=3)
#     % poly order 3
#     for i in range(columnsToPool):
#         for j in range(i, columnsToPool):
#             for k=j:nVars
#                 yout(:,colIdx) = inputArray(:,i).*inputArray(:,j).*inputArray(:,k);
#                 colIdx += 1;
#             end
#         end
#     end
# end

# if(polyorder>=4)
#     % poly order 4
#     for i in range(columnsToPool):
#         for j in range(i, columnsToPool):
#             for k=j:nVars
#                 for l=k:nVars
#                     yout(:,colIdx) = inputArray(:,i).*inputArray(:,j).*inputArray(:,k).*inputArray(:,l);
#                     colIdx += 1;
#                 end
#             end
#         end
#     end
# end

# if(polyorder>=5)
#     % poly order 5
#     for i in range(columnsToPool):
#         for j in range(i, columnsToPool):
#             for k=j:nVars
#                 for l=k:nVars
#                     for m=l:nVars
#                         yout(:,ind) = inputArray(:,i).*inputArray(:,j).*inputArray(:,k).*inputArray(:,l).*inputArray(:,m);
#                         ind = ind+1;
#                     end
#                 end
#             end
#         end
#     end
# end

# if(usesine)
#     for k=1:10;
#         yout = [yout sin(k*inputArray) cos(k*inputArray)];
#     end
# end