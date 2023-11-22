import timeit
import matplotlib.pyplot as plt

# code snippet to be executed only once
setup = """from multiprocessing import Process, Pipe
import random


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr.pop(random.randint(0, len(arr) - 1))
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


def quick_sort_parallel(arr, conn, proc_num):
    if proc_num <= 0 or len(arr) <= 1:
        conn.send(quick_sort(arr))
        conn.close()
        return

    pivot = arr.pop(random.randint(0, len(arr) - 1))

    left_side = [x for x in arr if x < pivot]
    right_side = [x for x in arr if x >= pivot]

    pconn_left, cconn_left = Pipe()
    left_proc = Process(target=quick_sort_parallel,
                        args=(left_side, cconn_left, proc_num - 1))

    pconn_right, cconn_right = Pipe()
    right_proc = Process(target=quick_sort_parallel,
                         args=(right_side, cconn_right, proc_num - 1))

    left_proc.start()
    right_proc.start()

    conn.send(pconn_left.recv() + [pivot] + pconn_right.recv())
    conn.close()

    left_proc.join()
    right_proc.join()
"""

solution_5_processes = """
if __name__ == '__main__':
    n = 2
    pconn, cconn = Pipe()
    
    p = Process(target=quick_sort_parallel,
                args=(arr, cconn, n))
    p.start()
    
    arr = pconn.recv()
"""

solution_10_processes = """
if __name__ == '__main__':
    n = 50
    pconn, cconn = Pipe()
    
    p = Process(target=quick_sort_parallel,
                args=(arr, cconn, n))
    p.start()
    
    arr = pconn.recv()
"""

code_qs_list = []
qs_5_time = []
qs_10_time = []
x = []

# maximum size of list
N = 250

# number of how many times code will be executed in one iteration
number = 50

for n in range(50, N):
    x.append(n)
    code_init = f"""
arr = [random.randint(0, 100) for x in range({n})]
"""
    code_qs_list.append(code_init)


for code in code_qs_list:
    time = timeit.timeit(setup=setup, stmt=code + solution_5_processes, number=number)
    qs_5_time.append(time)
    print('5 processes: ', time)


for code in code_qs_list:
    time = timeit.timeit(setup=setup, stmt=code + solution_10_processes, number=number)
    qs_10_time.append(time)
    print('10 processes: ', time)

plt.plot(x, qs_5_time, '-', label='5')
plt.plot(x, qs_10_time, '-', label='10')
plt.title("Comparison of 5 processes and 10 processes method")
plt.xlabel("Size of input array")
plt.ylabel("Time taken [seconds]")
plt.legend(loc="upper left")
plt.grid(True)

plt.show()
