import random


def bubble_sort(arr):
    n = len(arr)
    swapped = False
    for i in range(n -1):
        for j in range(0, n-i - 1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

        if not swapped:
            return


# Driver code to test above
arr = [64, 34, 25, 12, 22, 11, 90, 5, 2, 123]

bubble_sort(arr)

print("Sorted array is:")
for i in range(len(arr)):
    print("% d" % arr[i], end=" ")


def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min = i
        for j in range(i+1, n):
            if arr[j] <= arr[min]:
                min = j
        arr[i], arr[min] = arr[min], arr[i]


arr = [64, 34, 25, 12, 22, 11, 90, 5, 2, 123]

selection_sort(arr)

print("\nSorted array is:")
for i in range(len(arr)):
    print("% d" % arr[i], end=" ")
