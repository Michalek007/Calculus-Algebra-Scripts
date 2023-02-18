import numpy as np
import timeit

# get the LU decomposition of matrix A
def lu(A):
    # Get the number of rows
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    # Loop over rows
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, None] * U[i]
    return L, U


# get the Z matrix
def Z_matrix(B, L, U):
    n = B.shape[0]
    Z = np.zeros(n)
    for k in range(n):
        s = 0
        for p in range(k):
            s = s + (L[k][p] * Z[p])
        Z[k] = (B[k][0] - s) / L[k][k]
    return Z


# used to solve the last matrix equation where R is an upper triangular matrix
def back_substitution(R, y):
    n = R.shape[0]
    x = np.zeros(n)

    x[n - 1] = y[n - 1] / R[n - 1][n - 1]

    for k in range(n - 2, -1, -1):
        s = 0
        for p in range(n - 1, -1, -1):
            s = s + (R[k][p] * x[p])

        x[k] = (y[k] - s) / R[k][k]

    return x


# define the A matrix
A = np.array([[1, 1, 1],
              [1, 2, 3],
              [1.5, 2, 4]])

# define the B matrix
B = np.array([[1],
              [1],
              [1]])

# get the LU decomposition of the A matrix
L, U = lu(A)
# get the Z matrix
Z = Z_matrix(B, L, U)
# get the equation solution
X = back_substitution(U, Z)

print("A :")
print(A)

print("\nB :")
print(B)

print("\nL :")
print(L)

print("\nU :")
print(U)

print("\nZ :")
Z = Z[np.newaxis].transpose()
print(Z)

print("\nX matrix - the solution to our matrix equation:")
X = X[np.newaxis].transpose()
print(X)


# code snippet to be executed only once
setup = '''import numpy as np
#get the LU decomposition of matrix A
def lu(A):
    #Get the number of rows
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    #Loop over rows
    for i in range(n):    
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, None] * U[i]
    return L, U


#get the Z matrix
def Z_matrix(B, L, U):
    n = B.shape[0]
    Z = np.zeros(n)
    for k in range(n):
        s = 0
        for p in range(k):
            s = s + (L[k][p] * Z[p])
        Z[k] = (B[k][0] - s) / L[k][k] 
    return Z
    
#define the A matrix
A = np.array([[1,1,1],
              [1,2,3],
              [1.5,2,4]])

#define the B matrix
B = np.array([[1],
              [1],
              [1]])
              
'''

# code snippet whose execution time is to be measured
code = '''
#get the LU decomposition of the A matrix
L, U = lu(A)
#get the Z matrix
Z = Z_matrix(B, L, U)
#get the equation solution
X = np.linalg.inv(U).dot(Z)
'''

print('LU', timeit.timeit(setup=setup, stmt=code, number=10000))
