from multiprocessing import Process
import random


N = 100
arr = [random.randint(0, 100) for _ in range(N)]


def qs_parallel(arr, lo, hi):
    if lo >= hi or lo < 0:
        return
    p = partition(arr, lo, hi)
    proces1 = Process(target=qs_parallel, args=(arr, lo, p-1))
    proces2 = Process(target=qs_parallel, args=(arr, p+1, hi))


def partition(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi-1):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


qs_parallel(arr, 0, len(arr)-1)
print("\nSorted array is:")
for i in range(len(arr)):
    print("% d" % arr[i], end=" ")


N = 100
arr = [random.randint(0, 100) for _ in range(N)]


def qs(arr, lo, hi):
    if lo >= hi or lo < 0:
        return
    p = partition(arr, lo, hi)
    qs(arr, lo, p-1)
    qs(arr, p+1, hi)


qs_parallel(arr, 0, len(arr)-1)
print("\nSorted array is:")
for i in range(len(arr)):
    print("% d" % arr[i], end=" ")
