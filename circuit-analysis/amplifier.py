import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "Vin[V]",
                  y_label: str = "Vout[V]", continuous=True, r_signal=None, log_scale=False, axvline=None):
    mode = 0
    dpi = 300
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    if continuous:
        plt.plot(x, y, label='Punkty pomiarowe')
    else:
        plt.plot(x, y, '.', label='Punkty pomiarowe')
    if axvline:
        plt.axvline(axvline)
    if r_signal is not None:
        plt.plot(x, r_signal, label="Dopasowana funkcja")
        plt.legend(loc="upper left")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


df1 = pd.read_csv("data/data1.csv")
df2 = pd.read_csv("data/data2.csv")
df3 = pd.read_csv("data/data3.csv")


Vin1 = df1["Vin"]
Vout1 = df1["Vout"]

f2 = df2["f"]
k2 = df2["Vout"]/0.3


f3 = df3["f"]
k3 = df3["Vout"]/0.1

print("Wzmocnienie układu 2: ", k2[0])
print("Wzmocnienie układu 3: ", k3[1])


def func(x, a, b):
    return a*x + b

fit_params, cm = curve_fit(func, Vin1[4:16], Vout1[4:16])

display_graph(Vin1[4:16], Vout1[4:16], "Charakterystyka przejściowa (2)")
display_graph(Vin1, Vout1, "Charakterystyka przejściowa (1)")

# display_graph(Vin1[4:16], func(Vin1[4:16], fit_params[0], fit_params[1]), 'test', r_signal=Vout1[4:16])
print(fit_params)
print("Wzmocnienie układu 1: ", fit_params[0])

display_graph(f2, k2, "Charakterystyka częstotliwościowa (1)", y_label="k[V/V]", x_label="f[kHz]", log_scale=True, axvline=f2[13])
display_graph(f3, k3, "Charakterystyka częstotliwościowa (2)", y_label="k[V/V]", x_label="f[kHz]", log_scale=True, axvline=f3[14])

# print(20*np.log10(k2))
# print(20*np.log10(k3))

print("Częstotliwość graniczna układu 2 [kHz]: ", f2[13])
print("Częstotliwość graniczna układu 3 [kHz]: ", f3[14])

print("Pole wzmocnienia układu 2 [kHz]: ", k2[0] * f2[13])
print("Pole wzmocnienia układu 3 [kHz]: ", k3[1] * f3[14])
# print("fg6: ", f6[13])
# print("fg5: ", f5[5])
# print("fg4: ", f4[5])