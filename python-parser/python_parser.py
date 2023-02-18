import numpy as np
import matplotlib.pyplot as plt


with open('test.txt', 'r') as f:
    input_file = f.read().split('#')

code = []
for item in input_file:
    code.append("".join(item.split()))


special_char = ('.', '=', '(', ')')
math_char = ('*', '**', '-', '+', '/')

for item in code:
    for c in special_char:
        item = item.split(c)
    print(item)
    # for char in item:
    #     print(char)