import numpy as np
import math


def circuit_c(f):
    # data of circuit elements
    R1 = 4.7 * 10 ** 3
    R2 = 1 * 10 ** 3
    C1 = 220 * 10 ** -9
    C2 = 10 * 10 ** -9
    L1 = 10 * 10 ** -3
    E1 = 0.5

    w = 2 * np.pi * f

    # two nodes
    Y = np.array([
        [1j * w * C1 + 1 / R1 - 1j * 1 / (w * L1), 1j * 1 / (w * L1)],
        [1j * 1 / (w * L1), 1j * w * C2 + 1 / R2 - 1j * 1 / (w * L1)]
    ])
    I = np.array([E1 * 1j * w * C1, 0])

    W = np.linalg.det(Y)

    Y2 = np.array([
        [1j * w * C1 + 1 / R1 - 1j * 1 / (w * L1), 1j * 1 / (w * L1)],
        [E1 * 1j * w * C1, 0]
    ])

    W2 = np.linalg.det(Y2)

    # C2 voltage
    V2 = W2 / W

    I2 = V2 * C2 * w * 1j

    return np.abs(V2), np.angle(V2, deg=True), np.abs(I2), np.angle(I2, deg=True), W2, W


def circuit_b(f):
    # data of circuit elements
    R1 = 1 * 10 ** 3
    R2 = 1 * 10 ** 3
    C1 = 47 * 10 ** -9
    C2 = 100 * 10 ** -9
    C3 = 100 * 10 ** -9
    L1 = 10 * 10 ** -3
    E1 = 0.5

    w = 2 * np.pi * f

    # three nodes
    Y = np.array([
        [1j * w * C3 + 1 / R2 - 1j * 1 / (w * L1), 1j * 1 / (w * L1), -1 / R2],
        [1j * 1 / (w * L1), 1j * w * C2 + 1j * w * C1 - 1j * 1 / (w * L1), -1j * w * C1],
        [-1 / R2, -1j * w * C1, 1 / R1 + 1 / R2 + 1j * w * C1]
    ])
    I = np.array([0, E1 * 1j * w * C2, 0])

    W = np.linalg.det(Y)

    Y2 = np.array([
        [0, E1 * 1j * w * C2, 0],
        [1j * 1 / (w * L1), 1j * w * C2 + 1j * w * C1 + 1 / R2 - 1j * 1 / (w * L1), -1j * w * C1],
        [-1 / R2, -1j * w * C1, 1 / R1 + 1 / R2 + 1j * w * C1]
    ])

    W1 = np.linalg.det(Y2)

    # C1 voltage
    V1 = W1 / W

    I1 = V1 * C3 * w * 1j

    return np.abs(V1), np.angle(V1, deg=True), np.abs(I1), np.angle(I1, deg=True), W1, W


freq = (1000, 8500, 17000)

print('Circuit C')
for f in freq:
    print(
        f'Freq: {f}, V: {"%.5f" % circuit_c(f)[0]}, Phase: {"%.2f" % circuit_c(f)[1]}, I: {"%.6f" % circuit_c(f)[2]}, '
        f'Phase: {"%.2f" % circuit_c(f)[3]}, W2: {circuit_c(f)[4]}, W: {circuit_c(f)[5]}')

print('Circuit B')
for f in freq:
    print(
        f'Freq: {f}, V: {"%.5f" % circuit_b(f)[0]}, Phase: {"%.2f" % circuit_b(f)[1]}, I: {"%.6f" % circuit_b(f)[2]}, '
        f'Phase: {"%.2f" % circuit_b(f)[3]}, W2: {circuit_b(f)[4]}, W: {circuit_b(f)[5]}')


# def get_current(v, f):
#     C = 100 * 10 ** -9
#     w = 2 * np.pi * f
#     i = v * C * w * 1j
#     return np.abs(i)
#
#
# print("%.6f" % get_current(0.223, 1000))
# print("%.6f" % get_current(0.226, 9000))
# print("%.6f" % get_current(0.061, 17000))
#
# print("%.6f" % get_current(0.373, 1000))
# print("%.6f" % get_current(0.554, 9000))
# print("%.6f" % get_current(0.496, 17000))


def circuit_test(f):
    # data of circuit elements
    R1 = 2 * 10 ** 3
    R2 = 2 * 10 ** 3
    C1 = 10 * 10 ** -3
    C2 = 10 ** -6
    L1 = 1
    E1 = 1

    w = 2 * np.pi * f

    # print(1j * w * C1 + 1 / R1 - 1j * 1 / (w * L1))
    # print(1j * w * C2 + 1 / R2 - 1j * 1 / (w * L1))

    # two nodes
    Y = np.array([
        [1j * w * C1 + 1 / R1 - 1j * 1 / (w * L1), 1j * 1 / (w * L1)],
        [1j * 1 / (w * L1), 1j * w * C2 + 1 / R2 - 1j * 1 / (w * L1)]
    ])
    I = np.array([E1 * 1j * w * C1, 0])

    print(Y)
    print(I)

    W = np.linalg.det(Y)

    Y2 = np.array([
        [1j * w * C1 + 1 / R1 - 1j * 1 / (w * L1), 1j * 1 / (w * L1)],
        [E1 * 1j * w * C1, 0]
    ])

    W2 = np.linalg.det(Y2)

    # C2 voltage
    V2 = W2 / W

    I2 = V2 * C2 * w * 1j

    Y1 = np.array([
        [E1 * 1j * w * C1, 0],
        [1j * 1 / (w * L1), 1j * w * C2 + 1 / R2 - 1j * 1 / (w * L1)]
    ])

    W1 = np.linalg.det(Y1)

    V1 = W1 / W

    print(W1)
    print(np.abs(V1))

    print(np.abs(V1-V2))
    return np.abs(V2), np.angle(V2, deg=True), np.abs(I2), np.angle(I2, deg=True), W2, W


f = 100
results = circuit_test(f)
print('Test circuit: ')
print(f'Freq: {f}, V: {"%.5f" % results[0]}, Phase: {"%.2f" % results[1]}, I: {"%.6f" % results[2]}, '
      f'Phase: {"%.2f" % results[3]}, W2: {results[4]}, W: {results[5]}')

#
# print('________')
# for n in range(2, 10000):
#     print(n)
#     V = math.factorial(n)/math.factorial(n-2)
#     print(V)
#     print(V/2)
#     print('next')
#     if V % 2 != 0:
#         assert False
