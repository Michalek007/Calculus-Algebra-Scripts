import scipy
import numpy as np
import matplotlib.pyplot as plt
from math_functions import *


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, r_signal=None):
    # choose mode: 0 - show graphs, 1 - save as files
    mode = 1
    dpi = 300
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [V]")
    plt.title(title)
    plt.plot(x, y, label='Output')
    if r_signal is not None:
        plt.plot(x, r_signal, label='Input')
        plt.legend(loc="upper left")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


# amplitude, frequency and offset for input signal
freq = 100
amp = 1
offset = 1

# time domain <0, 1>
sampling = 44_000
t_lim = 2
time = np.linspace(-2, t_lim, sampling, endpoint=False)
N = len(time)


# display_graph(time, PI(time), 'PI(t)')

def b1(x):
    return pi(x-0.5)


def b2(x):
    return b1(2*x) - b1(2*x-1)


def b3(x):
    return b2(2*x) - b2(2*x-1)


def b4(x):
    return b2(2*x) + b2(2*x-1)


def walsh_base(x):
    b1 = pi(x - 0.5)
    b2 = pi(2*x-0.5) - pi(2*x-1.5)
    b3 = pi(4*x-0.5) - pi(4*x-1.5) - pi(4*x-2.5) + pi(4*x-3.5)
    b4 = pi(4 * x - 0.5) - pi(4 * x - 1.5) + pi(4 * x - 2.5) - pi(4 * x - 3.5)
    return (b1, "b1(t)"), (b2, "b2(t)"), (b3, "b3(t)"), (b4, "b4(t)")
# b1 = pi(time-0.5)
# b2 = pi(time-0.5) - 2 * pi(time-1.5)
# b3 = pi(time-0.5) - 2 * pi(4*(time-0.5))
# b4 = pi(time-0.5) - 2 * pi(8*(time-0.375)) - 2 * pi(8*(time-0.875))


display_graph(time, b1(time/4 + 0.5), 'b1(t)')
display_graph(time, b2(time/4 + 0.5), 'b2(t)')
display_graph(time, b3(time/4 + 0.5), 'b3(t)')
display_graph(time, b4(time/4 + 0.5), 'b4(t)')

x_apr = 1/4 * b1(time/4 + 0.5) - 0.5 * b2(time/4 + 0.5) - 0.5 * b3(time/4 + 0.5)
display_graph(time, pi(time/2)-tri(time+1), "main_fun", r_signal=x_apr-0.25)
display_graph(time, x_apr, "x_apr(t)")


# for b, title in walsh_base(time):
#     display_graph(time, b, title)
