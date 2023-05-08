import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "Odległość",
                  y_label: str = "Czas trwania impulsu", continuous=True, r_signal=None):
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
        plt.plot(x, y, '.-', label='Punkty pomiarowe')
    else:
        plt.plot(x, y, '.', label='Punkty pomiarowe')
    if r_signal is not None:
        plt.plot(x, r_signal, label="Dopasowana funkcja")
        plt.legend(loc="upper left")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


df2 = pd.read_csv("../data/dalmierz2.csv")
distance = df2["distance"]
optical = df2["optical"]
sound = df2["sound"]
impuls = df2["impuls"]
velocity = df2["velocity"]

display_graph(distance, impuls, "Ultrasound sensor")

df2 = pd.read_csv("../data/dalmierz3.csv")
distance_lin = df2["distance"]
impuls_lin = df2["impuls"]
display_graph(distance_lin, impuls_lin, "Ultrasound sensor - linear range")


def func(x, a, b):  # nasza funkcja fitowania
    return a*x+b


fit_params, matrix = curve_fit(func, distance_lin, impuls_lin)
print(f"Czułość: {fit_params[0]} Offset {fit_params[1]}")
print(f"Zakres martwy: {750}")


display_graph(distance, optical, "Optical sensor", y_label="Distance measured by optical sensor", x_label="Real distance")
fit_params, matrix = curve_fit(func, distance, optical)
print(f"Czułość: {fit_params[0]} Offset {fit_params[1]}")
print(f"Zakres martwy: {1300}")
