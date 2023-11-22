import functools
import timeit
import matplotlib.pyplot as plt

# code snippet to be executed only once
setup_cached = """import functools
@functools.lru_cache(maxsize=10000)
def calculate(n):
    if n < 2:
        return n
    else:
        return calculate(n-2) + calculate(n-1)
"""

setup = """def calculate(n):
    if n < 2:
        return n
    else:
        return calculate(n-2) + calculate(n-1)
"""

solution = """
fibonacci = calculate(num)
"""

code_list = []
time_list = []
time_list_cashed = []
x = []

# number of max generated fibonacci sequence
N = 25

# number of how many times code will be executed in one iteration
number = 100
for n in range(9, N):
    x.append(n)
    code_init = f"""
num = {n}
"""
    code_list.append(code_init)

for code in code_list:
    time = timeit.timeit(setup=setup, stmt=code + solution, number=number)
    time_list.append(time)
    print(time)

for code in code_list:
    time = timeit.timeit(setup=setup_cached, stmt=code + solution, number=number)
    time_list_cashed.append(time)
    print(time)


plt.plot(x, time_list, '-', label='without lru_cache')
plt.plot(x, time_list_cashed, '-', label='with lru_cache')
plt.title("Execution time of recursive fibonacci sequence generator")
plt.xlabel("Generated number of sequence")
plt.ylabel("Time taken [seconds]")
plt.legend(loc="upper left")
plt.grid(True)

plt.show()
