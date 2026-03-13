import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from fonctions_annexes import ic_print, ic_pivot_print, ic_pivot, ic_pivot2

def ic_print(A,eta):
    n=len(A)
    M=np.matrix.trace(A)

    L=np.zeros((n,n)) #à changer pour éviter le O(n^2)
    L[0,0]=sqrt(A[0,0])
    for j in range (1,n):
        L[j,0]=A[0,j]/L[0,0]
        print(L)
    S=sum(L[j,0]**2 for j in range(n)) 
    i=1       
    
    while S<(1-eta)*M: #écrit d'une manière différente dans le papier de recherche : ||C - GG'||                       #1
        s1 = sum(L[i,k]**2 for k in range(i))
        L[i,i]=sqrt(A[i,i]-s1)
        print(L)
        
        for j in range(i+1,n):      #2                   
            s2 = sum(L[i,k]*L[j,k] for k in range(i)) #3
            L[j,i]=(A[i,j]-s2)/L[i,i] 
            print(L)
            
        S+=sum(L[j,i]**2 for j in range(n)) #norme euclidienne au carrée de la nouvelle colonne
        i+=1
    #print(S)
    print(L[::,:i])
    

def max_gauche(tab):
  maxval = tab[0][0]
  maxind = 0
  for i in range(1,len(tab)):
    if tab[i][0] > maxval:
      maxval = tab[i][0]
      maxind = i
  return tab[maxind],maxind


def ic_pivot_print(A,eta): #Visualisation de cholesky incomplet avec pivot
    n=len(A)
    M=np.matrix.trace(A)
    D = np.array([(A[i,i],i) for i in range(n)]) #Tableau des bornes inf : L[ik,k]^2 = D(k-1)[ik]
    I = []
    J = np.array([i for i in range(n)])
    # à noter : new_tab = np.delete(tab,[indexes])
    L=np.zeros((n,n)) #à changer pour éviter le O(n^2)
    
    S=0
    k=0
    
    
    while S<(1-eta)*M: 
      
        print("\n----------------NOUVELLE COLONNE!----------------\n")
        (borne_inf_max,ik),ind_delete = max_gauche(D) #on détermine le prochain pivot avec la borne inf
        print("on choisit comme pivot l'indice ",int(ik))
        I.append(int(ik))
        D = np.delete(D,[ind_delete],axis=0) # on met à jour les tableaux D,I,J : ind_delete existe car i peut très bien être à une autre place place que celle d'indice i dans les tableaux
        J = np.delete(J,[ind_delete])
        print("indices des colonnes inserees dans l'ordre chrono : ",I)
        print("\nJ : ",J)
        L[int(ik),k] = sqrt(borne_inf_max) #ajout du "coeff diagonal"
        
        print("\n!ajout du coeff diagonal!\n")
        print("Matrice L :\n",L)
        print("\n !remplissage du reste de la colonne! \n")
        
        for e in J:
          L[e,k] = (A[e,int(ik)] - sum(L[e][j]*L[int(ik)][j] for j in range(k)) )/(L[int(ik),k]) #ajout de la colonne
          print("Matrice L :\n",L)
          print("")
        
        for i in range(n-k-1):
          D[i][0] -= L[J[i],k]**2 #mise à jour des gains possibles (se justifie par l'évaluation des coeffs diag du produit matriciel A = LL.T)
      
        S += sum(L[i,k]**2 for i in range(n))
        k+=1
    print("sous-matrice triangulaire: \n",L[I,:k]) #L'ordre dans lequel les indices sont ajoutés dans I est important
 


size = 5
A = 2 * np.random.rand(size, size)
matrice = A.T @ A

M = np.array([[88, -48,  60, -16],
              [-48, 36, -20,   8],
              [60, -20, 76, -28],
              [-16,8,-28,40]])


print(ic_pivot(matrice,0.05))
print(ic_pivot2(matrice,0.05))