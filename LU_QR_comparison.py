import timeit
import matplotlib.pyplot as plt

# code snippet to be executed only once
setup_LU = '''import numpy as np
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
'''

# code snippet to be executed only once
setup_QR = '''import numpy as np
#householder decomposition to Q and R matrices
def qr(A):
    n = A.shape[0]
    Q = np.eye(n)

    for i in range(n-1):
        H = np.eye(n)
        H[i:, i:] = make_householder(A[i:, i])
        #Q = QH
        Q = Q @ H
        #R = HR
        A = H @ A

    return Q, A

#returns a householder matrix
def make_householder(A):
    u = A / (A[0] + np.copysign(np.linalg.norm(A), A[0]))
    u[0] = 1
    H = np.eye(A.shape[0])

    H = H - (2 / np.dot(u, u)) * np.transpose(u[np.newaxis]) @ u[np.newaxis]

    return H

#used to solve the last matrix equation where R is an upper triangular matrix
def back_substitution(R, y):
    n = R.shape[0]
    x = np.zeros(n)

    x[n-1] = y[n-1] / R[n-1][n-1]

    for k in range(n-2, -1, -1):
        s = 0
        for p in range(n-1, -1, -1):
            s = s + (R[k][p] * x[p])

        x[k] = (y[k] - s) / R[k][k]

    return x


#function that solves the equation QRx = B
def solve_qr(Q, R, B):
    #QRx = B
    Q = np.transpose(Q)
    #R * x = Qt * B
    Y = Q @ B
    X = back_substitution(R, Y)

    return X
'''

solution_QR = '''
#solve the equation
Q, R = qr(A)
X = solve_qr(Q, R, B)
'''

solution_LU = '''
#get the LU decomposition of the A matrix
L, U = lu(A)
#get the Z matrix
Z = Z_matrix(B, L, U)
#get the equation solution
X = np.linalg.inv(U).dot(Z)
'''

code_QR_list = []
code_LU_list = []
QR_time = []
LU_time = []
x = []

# maximum size of matrix
N = 250

# numer of how many times code will be executed in one iteration
number = 50

for n in range(3, N):
    x.append(n)
    code_QR = f'''
#define the A matrix
A = np.random.rand({n}, {n})

#define the B matrix
B = np.ones(({n}, 1))
'''
    code_QR_list.append(code_QR)

for code in code_QR_list:
    time = timeit.timeit(setup=setup_QR, stmt=code + solution_QR, number=number)
    QR_time.append(time)
    print('QR: ', time)


for n in range(3, N):
    code_LU = f'''
#define the A matrix
A = np.random.rand({n}, {n})

#define the B matrix
B = np.ones(({n}, 1))
'''
    code_LU_list.append(code_LU)

for code in code_LU_list:
    time = timeit.timeit(setup=setup_LU, stmt=code + solution_LU, number=number)
    LU_time.append(time)
    print('LU: ', time)

plt.plot(x, LU_time, '.', label='LU')
plt.plot(x, QR_time, '.', label='QR')
plt.title("Comparison of LU and QR method")
plt.xlabel("Size of matrix")
plt.ylabel("Time taken [seconds]")
plt.legend(loc="upper left")
plt.grid(True)

plt.show()
