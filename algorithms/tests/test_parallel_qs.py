import unittest
from multiprocessing import Pipe, Process

from algorithms.scripts.parallel_qs import quick_sort_parallel


class TestQsParallel(unittest.TestCase):

    def test_qs_4_3_processes(self):
        arr = [24, 31, 56, 23]
        n = 3  # 2**(n+1) - 1 processes will be instantiated.
        pconn, cconn = Pipe()

        p = Process(target=quick_sort_parallel,
                    args=(arr, cconn, n))
        p.start()

        returned_arr = pconn.recv()
        expected_arr = [23, 24, 31, 56]
        self.assertEqual(returned_arr, expected_arr)

    def test_qs_100_3_processes(self):
        arr = [13, 38, 93, 50, 81, 16, 36, 30, 29, 70, 90, 10, 80, 84, 88, 88, 2, 62, 82, 60, 63, 66, 64, 76, 66, 43, 85, 61, 24, 24, 98, 91, 9, 71, 18, 6, 47, 11, 26, 37, 58, 77, 61, 74, 23, 37, 82, 49, 9, 41, 96, 29, 4, 81, 20, 59, 14, 46, 76, 29, 59, 12, 50, 20, 18, 21, 42, 17, 24, 7, 30, 33, 25, 5, 97, 65, 66, 40, 64, 42, 14, 16, 71, 66, 33, 24, 92, 71, 1, 78, 73, 93, 69, 54, 33, 47, 68, 11, 13, 100]
        n = 3 # 2**(n+1) - 1 processes will be instantiated.
        pconn, cconn = Pipe()

        p = Process(target=quick_sort_parallel,
                    args=(arr, cconn, n))
        p.start()

        returned_arr = pconn.recv()
        expected_arr = [1, 2, 4, 5, 6, 7, 9, 9, 10, 11, 11, 12, 13, 13, 14, 14, 16, 16, 17, 18, 18, 20, 20, 21, 23, 24, 24, 24, 24, 25, 26, 29, 29, 29, 30, 30, 33, 33, 33, 36, 37, 37, 38, 40, 41, 42, 42, 43, 46, 47, 47, 49, 50, 50, 54, 58, 59, 59, 60, 61, 61, 62, 63, 64, 64, 65, 66, 66, 66, 66, 68, 69, 70, 71, 71, 71, 73, 74, 76, 76, 77, 78, 80, 81, 81, 82, 82, 84, 85, 88, 88, 90, 91, 92, 93, 93, 96, 97, 98, 100]
        self.assertEqual(returned_arr, expected_arr)

    def test_qs_100_5_process(self):
        arr = [13, 38, 93, 50, 81, 16, 36, 30, 29, 70, 90, 10, 80, 84, 88, 88, 2, 62, 82, 60, 63, 66, 64, 76, 66, 43, 85, 61, 24, 24, 98, 91, 9, 71, 18, 6, 47, 11, 26, 37, 58, 77, 61, 74, 23, 37, 82, 49, 9, 41, 96, 29, 4, 81, 20, 59, 14, 46, 76, 29, 59, 12, 50, 20, 18, 21, 42, 17, 24, 7, 30, 33, 25, 5, 97, 65, 66, 40, 64, 42, 14, 16, 71, 66, 33, 24, 92, 71, 1, 78, 73, 93, 69, 54, 33, 47, 68, 11, 13, 100]
        n = 5 # 2**(n+1) - 1 processes will be instantiated.
        pconn, cconn = Pipe()

        p = Process(target=quick_sort_parallel,
                    args=(arr, cconn, n))
        p.start()

        returned_arr = pconn.recv()
        expected_arr = [1, 2, 4, 5, 6, 7, 9, 9, 10, 11, 11, 12, 13, 13, 14, 14, 16, 16, 17, 18, 18, 20, 20, 21, 23, 24, 24, 24, 24, 25, 26, 29, 29, 29, 30, 30, 33, 33, 33, 36, 37, 37, 38, 40, 41, 42, 42, 43, 46, 47, 47, 49, 50, 50, 54, 58, 59, 59, 60, 61, 61, 62, 63, 64, 64, 65, 66, 66, 66, 66, 68, 69, 70, 71, 71, 71, 73, 74, 76, 76, 77, 78, 80, 81, 81, 82, 82, 84, 85, 88, 88, 90, 91, 92, 93, 93, 96, 97, 98, 100]
        self.assertEqual(returned_arr, expected_arr)


if __name__ == '__main__':
    unittest.main()
