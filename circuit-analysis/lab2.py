import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "Vin[V]",
                  y_label: str = "Vout[V]", continuous=True, r_signal=None, log_scale=False):
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
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    if continuous:
        plt.plot(x, y, '.-', label='Punkty pomiarowe')
    else:
        plt.plot(x, y, '.', label='Punkty pomiarowe')
    if r_signal is not None:
        r_signal.reverse()
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



# def func(x, a):
#     return a*x
#
#
display_graph(Vin1, Vout1, "Charakterystyka przejściowa wzmacniacza odwracającego")
# fit_params, cm = curve_fit(func, Vin2, Vout2)
# print(fit_params)
#
# display_graph(Vin3, Vout3, "Charakterystyka przejściowa wzmacniacza odwracającego z buforem")
# fit_params, cm = curve_fit(func, Vin3, Vout3)
# print(fit_params)
#
display_graph(f2, k2, "Charakterystyka częstotliwościowa wzmacniacza odwracającego", y_label="k[V/V]", x_label="f[kHz]", log_scale=True)
display_graph(f3, k3, "Charakterystyka częstotliwościowa", y_label="k[V/V]", x_label="f[kHz]", log_scale=True)
# display_graph(VinS, VoutS, "Charakterystyka przejściowa - wzmocnienie sumacyjne", y_lim=[0, 0.1])
# display_graph(VinR, VoutR, "Charakterystyka przejściowa - wzmocnienie różnicowe")
#
# # print(20*np.log10(k6))
# # print(20*np.log10(k5))
# # print(20*np.log10(k4))
#
# print("fg6: ", f6[13])
# print("fg5: ", f5[5])
# print("fg4: ", f4[5])
#
# print(VinR[-1]-VoutR[0])