import numpy as np
import timeit


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



#define the A matrix
A = np.array([[1,1,1],
              [1,2,3],
              [1.5,2,4]])

#define the B matrix
B = np.array([[1],
              [1],
              [1]])

#solve the equation
Q, R = qr(A)
X = solve_qr(Q, R, B)

print("A :")
print(A)
print("\nB :")
print(B)
print("\nR : ")
print(R)
print("\nQ : ")
print(Q)
print("\nX - solution : ")
print(np.transpose(X[np.newaxis]))


# code snippet to be executed only once
mysetup = '''import numpy as np
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
mycode = '''
#solve the equation
Q, R = qr(A)
X = solve_qr(Q, R, B)
'''


print('QR', timeit.timeit(setup=mysetup, stmt=mycode, number=10000))


# code snippet whose execution time is to be measured
code_QR_3x3 = '''
#define the A matrix
A = np.array([[1,1,1],
              [1,2,3],
              [1.5,2,4]])

#define the B matrix
B = np.array([[1],
              [1],
              [1]])
'''

code_QR_4x4 = '''
#define the A matrix
A = np.array([[1,1,1,1],
              [1,2,3,4],
              [1.5,2,4,5],
              [1.5,2,4,5]])

#define the B matrix
B = np.array([[1],
              [1],
              [1],
              [1]])
'''

code_QR_5x5 = '''
#define the A matrix
A = np.array([[1,1,1,1,1],
              [1,2,3,4,5],
              [1.5,2,4,5,6],
              [1.5,2,4,5,6],
              [1.5,2,4,5,6]])

#define the B matrix
B = np.array([[1],
              [1],
              [1],
              [1],
              [1]])
'''

code_QR_6x6 = '''
#define the A matrix
A = np.array([[1,1,1,1,1,1],
              [1,2,3,4,5,6],
              [1,2,4,5,6,8],
              [1,2,3,4,5,6],
              [1,2,3,4,5,6],
              [1,2,3,4,5,6]])

#define the B matrix
B = np.array([[1],
              [1],
              [1],
              [1],
              [1],
              [1]])
'''

code_QR_7x7 = '''
#define the A matrix
A = np.ones((7, 7)) * random.randint(1,5)
#define the B matrix
B = np.ones((7, 1))
'''

code_QR_8x8 = '''
#define the A matrix
A = np.ones((8, 8)) * random.randint(1,5)
#define the B matrix
B = np.ones((8, 1))
'''

code_QR_9x9 = '''
#define the A matrix
A = np.ones((9, 9)) * random.randint(1,5)
#define the B matrix
B = np.ones((9, 10))
'''

code_QR_10x10 = '''
#define the A matrix
A = np.ones((10, 10)) * random.randint(1,5)
#define the B matrix
B = np.ones((10, 1))
'''



# print('Calculation time\n')
# for i in range(1, 6):
#     print(f'Number of repetition: {i*10000}')
#     print('LU: ', timeit.timeit(setup=setup_LU, stmt=code_LU, number=i*10000))
#     print('QR: ', timeit.timeit(setup=setup_QR, stmt=code_QR, number=i*10000))
#     print('\n')


# print('Calculation time\n')
# for i in range(1, 6):
#     print(f'Number of repetition: {i*10000}')
#     print('LU: ', timeit.timeit(setup=setup_LU, stmt=code_LU, number=i*10000))
#     print('QR: ', timeit.timeit(setup=setup_QR, stmt=code_QR, number=i*10000))
#     print('\n')


# code_QR_tuple = (code_QR_3x3, code_QR_4x4, code_QR_5x5, code_QR_6x6, code_QR_7x7, code_QR_8x8, code_QR_9x9, code_QR_10x10)

# for code in code_QR:
#     print('QR: ', timeit.timeit(setup=setup_QR, stmt=code + solution_QR, number=10000))


import random
import numpy as np