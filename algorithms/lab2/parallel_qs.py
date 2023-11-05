from multiprocessing import Process, Pipe
import random


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr.pop(random.randint(0, len(arr) - 1))
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


def quick_sort_parallel(arr, conn, proc_num):
    """
    Partition the list, then quicksort the left and right
    sides in parallel.
    """

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


def main():
    N = 100

    arr = [random.randint(0, 100) for x in range(N)]

    print("Unsorted Array")
    print(arr)

    n = 1  # 2**(n+1) - 1 processes will be instantiated.
    pconn, cconn = Pipe()

    p = Process(target=quick_sort_parallel,
                args=(arr, cconn, n))
    p.start()

    arr = pconn.recv()
    print('Sorted Array in Ascending Order:')
    print(arr)


if __name__ == '__main__':
    main()
