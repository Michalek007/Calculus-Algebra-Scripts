
class Fibonacci:
    def __init__(self, steps: int):
        self.steps = steps

    def __iter__(self):
        self.value = 0
        self.iteration = 0
        return self

    def __next__(self):
        self.value = self.calculate(self.iteration)
        self.iteration += 1
        if self.iteration > self.steps:
            raise StopIteration
        return self.value

    def calculate(self, n):
        if n < 2:
            return n
        else:
            return self.calculate(n-2) + self.calculate(n-1)


if __name__ == '__main__':
    # x = iter(Fibonacci(3))
    # print(next(x))

    for item in Fibonacci(20):
        print(item)
