import scipy
import numpy as np
import matplotlib.pyplot as plt


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, r_signal=None):
    # choose mode: 0 - show graphs, 1 - save as files
    mode = 0
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


def PI(t):
    n = len(t)
    y = np.zeros(n)
    for i in range(n-1):
        if 1 > t[i] > -1:
            y[i] = 1
    return y


# amplitude, frequency and offset for input signal
freq = 100
amp = 1
offset = 1

# time domain <0, 1>
sampling = 44_000
t_lim = 1
time = np.linspace(0, t_lim, sampling, endpoint=False)
N = len(time)


# display_graph(time, PI(time), 'PI(t)')

b1 = PI(time-0.5)
b2 = PI(time-0.5) - 2 * PI(time-1.5)
b3 = PI(time-0.5) - 2 * PI(4*(time-0.5))
b4 = PI(time-0.5) - 2 * PI(8*(time-0.375)) - 2 * PI(8*(time-0.875))

display_graph(time, b1, 'b1(t)')
display_graph(time, b2, 'b2(t)')
display_graph(time, b3, 'b3(t)')
display_graph(time, b4, 'b3(t)')
