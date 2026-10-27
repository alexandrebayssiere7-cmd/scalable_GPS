import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from helpers import ic_print, ic_pivot_print, ic_pivot, ic_pivot2

def ic_print(A,eta):
    n=len(A)
    M=np.matrix.trace(A)

    L=np.zeros((n,n)) # to change to avoid the O(n^2)
    L[0,0]=sqrt(A[0,0])
    for j in range (1,n):
        L[j,0]=A[0,j]/L[0,0]
        print(L)
    S=sum(L[j,0]**2 for j in range(n))
    i=1

    while S<(1-eta)*M: # written differently in the research paper: ||C - GG'||                       #1
        s1 = sum(L[i,k]**2 for k in range(i))
        L[i,i]=sqrt(A[i,i]-s1)
        print(L)

        for j in range(i+1,n):      #2
            s2 = sum(L[i,k]*L[j,k] for k in range(i)) #3
            L[j,i]=(A[i,j]-s2)/L[i,i]
            print(L)

        S+=sum(L[j,i]**2 for j in range(n)) # squared Euclidean norm of the new column
        i+=1
    #print(S)
    print(L[::,:i])


def max_first_column(tab):
  maxval = tab[0][0]
  maxind = 0
  for i in range(1,len(tab)):
    if tab[i][0] > maxval:
      maxval = tab[i][0]
      maxind = i
  return tab[maxind],maxind


def ic_pivot_print(A,eta): # Step-by-step view of the pivoted incomplete Cholesky
    n=len(A)
    M=np.matrix.trace(A)
    D = np.array([(A[i,i],i) for i in range(n)]) # Array of lower bounds: L[ik,k]^2 = D(k-1)[ik]
    I = []
    J = np.array([i for i in range(n)])
    # note: new_tab = np.delete(tab,[indexes])
    L=np.zeros((n,n)) # to change to avoid the O(n^2)

    S=0
    k=0


    while S<(1-eta)*M:

        print("\n----------------NEW COLUMN!----------------\n")
        (max_lower_bound,ik),ind_delete = max_first_column(D) # pick the next pivot from the lower bound
        print("pivot chosen: index ",int(ik))
        I.append(int(ik))
        D = np.delete(D,[ind_delete],axis=0) # update the arrays D,I,J: ind_delete is needed because i may sit at a position other than i in the arrays
        J = np.delete(J,[ind_delete])
        print("indices of inserted columns in chronological order: ",I)
        print("\nJ : ",J)
        L[int(ik),k] = sqrt(max_lower_bound) # add the "diagonal coefficient"

        print("\n!diagonal coefficient added!\n")
        print("Matrix L:\n",L)
        print("\n !filling the rest of the column! \n")

        for e in J:
          L[e,k] = (A[e,int(ik)] - sum(L[e][j]*L[int(ik)][j] for j in range(k)) )/(L[int(ik),k]) # add the column
          print("Matrix L:\n",L)
          print("")

        for i in range(n-k-1):
          D[i][0] -= L[J[i],k]**2 # update the possible gains (justified by the diagonal coefficients of the matrix product A = LL.T)

        S += sum(L[i,k]**2 for i in range(n))
        k+=1
    print("triangular submatrix: \n",L[I,:k]) # The order in which indices are added to I matters



size = 5
A = 2 * np.random.rand(size, size)
matrix = A.T @ A

M = np.array([[88, -48,  60, -16],
              [-48, 36, -20,   8],
              [60, -20, 76, -28],
              [-16,8,-28,40]])


print(ic_pivot(matrix,0.05))
print(ic_pivot2(matrix,0.05))