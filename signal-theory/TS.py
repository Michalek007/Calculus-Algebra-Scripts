import numpy as np
import scipy
import matplotlib.pyplot as plt


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "Frequency [Hz]",
                  y_label: str = "Amplitude", continuous=True, r_signal=None):
    mode = 1
    dpi = 300
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if continuous:
        plt.plot(x, y, '.-', label='Output')
    else:
        plt.plot(x, y, '.', label='Output')
    if r_signal is not None:
        plt.plot(x, r_signal, label='Input')
        plt.legend(loc="upper left")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


def pi(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        if -0.5 < x[i] < 0.5:
            y[i] = 1
        elif x[i] == -0.5 or x[i] == 0.5:
            y[i] = 0.5
    return y


def tri(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        if abs(x[i]) < 1:
            y[i] = 1 - abs(x[i])
    return y


def sinc(x):
    n = len(x)
    y = np.zeros(n)
    sin = np.sin(x)
    for i in range(n):
        if x[i] != 0:
            y[i] = sin[i]/x[i]
        else:
            y[i] = 1
    return y

sampling = 44_000
t_lim = 5
time = np.linspace(-t_lim, t_lim, sampling, endpoint=False)
N = len(time)

# time domain <-1, 1>
# sampling = 88_000
# t_lim = 5
# time_sinc = np.linspace(-t_lim, t_lim, sampling, endpoint=False)

f1 = 4000
x_t = pi(f1 * time) * np.sin(2 * np.pi * time * (f1/3))
display_graph(time, x_t, "x(t)", x_lim=[-0.0005, 0.0005])
# display_graph(time_sinc, sinc(time_sinc * np.pi), "Sinc(x)", r_signal=sinc(time_sinc))
# display_graph(time, -tri((time+2)/6), 'Triangle1')
# display_graph(time, tri((time-2)/6), 'Triangle2')
display_graph(time, tri((time+1)/4) + tri((time-1)/4), 'Triangle+')

# sinc sum equals 1
# lim = 200
# result = np.zeros(N)
# for n in range(-lim, lim):
#     result += sinc((time*np.pi/f1) - n * f1)
#
# print(np.max(result))
# print(np.min(result))
# display_graph(time, result, "Xp(f)", y_lim=[-2, 2])

display_graph(time, sinc(np.pi * time/5), "Skalowanie")
display_graph(time, sinc((np.pi * time) - 2), "Przesuniecie")
