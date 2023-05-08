import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "H_measured[G]",
                  y_label: str = "Crosstalk", continuous=True, r_signal=None):
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
        r_signal.reverse()
        plt.plot(x, r_signal, label="Dopasowana funkcja")
        plt.legend(loc="upper left")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


df2 = pd.read_csv("../data/Data.csv")
H_row_x = df2["H_row_x"]/230
H_row_y = df2["H_row_y"]/230
H_row_z = df2["H_row_z"]/230
H_set = df2["H_set"]
H_rowInv = list(df2["H_rowInv"]/230)
H_rowInv_y = list(df2["H_rowInv_y"]/230)
H_rowInv_z = list(df2["H_rowInv_z"]/230)
Voltage = df2["Voltage"]
Current = df2["Current"]

display_graph(H_set, H_row_x, "H_measured vs H_set - X axis", r_signal=H_rowInv)
display_graph(H_set, H_row_y, "H_measured vs H_set - Y axis", r_signal=H_rowInv_y)
display_graph(H_set, H_row_z, "H_measured vs H_set - Z axis", r_signal=H_rowInv_z)

df2 = pd.read_csv("../data/Data2.csv")
H_row_x_lin = list(df2["H_row_x"]/230)
H_row_y_lin = list(df2["H_row_y"]/230)
H_row_z_lin = list(df2["H_row_z"]/230)
H_rowInv_lin = list(df2["H_rowInv"]/230)
H_rowInv_y_lin = list(df2["H_rowInv_y"]/230)
H_rowInv_z_lin = list(df2["H_rowInv_z"]/230)

H_set_lin = df2["H_set"]


def func(x, a, b):  # nasza funkcja fitowania
    return a*x+b


fit_params, covariance_matrix = curve_fit(func, H_set_lin, H_row_x_lin)  # fit_params - parametry futowania, covariance_matrix - macierz kowariancji (nie używmay jej w tym ćwieczeniu)
print(fit_params)
display_graph(H_set_lin, H_row_x_lin, "Dopasowanie liniowe funkcji - X", r_signal=func(H_set_lin, *fit_params), continuous=False)
fit_params, covariance_matrix = curve_fit(func, H_set_lin, H_row_y_lin)
print(fit_params)
display_graph(H_set_lin, H_row_y_lin, "Dopasowanie liniowe funkcji - Y", r_signal=func(H_set_lin, *fit_params), continuous=False)
fit_params, covariance_matrix = curve_fit(func, H_set_lin, H_row_z_lin)
print(fit_params)
display_graph(H_set_lin, H_row_z_lin, "Dopasowanie liniowe funkcji - Z", r_signal=func(H_set_lin, *fit_params), continuous=False)


crosstalk_y = [H_row_y_lin[i] / H_row_x_lin[i] for i in range(len(H_row_y_lin))]
crosstalk_z = [H_row_z_lin[i] / H_row_x_lin[i] for i in range(len(H_row_y_lin))]

display_graph(H_row_y_lin, crosstalk_y, "Crosstalk - Y")
display_graph(H_row_z_lin, crosstalk_z, "Crosstalk - Z")

# display_graph(H_set_lin, H_row_x_lin, "H_measured vs H_set - X axis", r_signal=H_rowInv_lin)
# display_graph(H_set_lin, H_row_y_lin, "H_measured vs H_set - Y axis", r_signal=H_rowInv_y_lin)
# display_graph(H_set_lin, H_row_z_lin, "H_measured vs H_set - Z axis", r_signal=H_rowInv_z_lin)

