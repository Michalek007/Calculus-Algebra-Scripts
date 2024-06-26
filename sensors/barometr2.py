import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import scipy.fft
from scipy.optimize import curve_fit
import math


def create_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = None,
                 y_label: str = None, label: list = None, continuous=True, x_ticks: list = None,
                 y_ticks: list = None, values: list[list] = None):
    mode = 0
    dpi = 150
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.grid()
    plt.title(title)
    # plt.axis('off')
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if x_ticks:
        plt.xticks(x_ticks)
    if y_ticks:
        plt.yticks(y_ticks)
    if continuous:
        plt.plot(x, y, label=label[0] if label else None)
    else:
        plt.plot(x, y, '.', label=label[0] if label else None)
    if values:
        for i in range(len(values)):
            plt.plot(values[i][0], values[i][1], label=label[i+1] if label else None)
        plt.legend(loc="upper right")
    if mode:
        plt.savefig(title + '.png', dpi=dpi)
        plt.clf()
    else:
        plt.show()


def equal(a, b, e=0.01):
    return b + e >= a >= b - e


def derivative(y, x):
    diff_x = []
    for i in range(len(x)):
        try:
            diff_x.append((y[i+1] - y[i]) / (x[i+1] - x[i]))
        except IndexError:
            break
    return diff_x


# akcelerometr
with open("data/pomiar1.txt", 'r') as f:
    data = f.read().split('\n')

time = []
y_axis = []
for row in data:
    row_data = row.split("\t")
    if len(row_data) < 4:
        time.append(row_data[0].split(" ")[0])
        time[-1] = float(time[-1])
        # x_axis[-1] = float(x_axis[-1])
        y_axis.append(float(row_data[1]))
    else:
        time.append(float(row_data[0]))
        y_axis.append(float(row_data[2]))

harmonic = []
harmonic_time = []
for i in range(int(30/0.05), int(78/0.05)):
    harmonic.append(y_axis[i])
    harmonic_time.append(time[i])


average = abs(np.sum(harmonic)/len(harmonic))
print("Average acceleration: ", average)
create_graph(harmonic_time, harmonic, "Acceleration vs time", x_label="Time [s]", y_label="Acceleration [m/s^2]")

c_harmonic = []
for i in range(len(harmonic)):
    c_harmonic.append(harmonic[i] + average)

create_graph(harmonic_time, c_harmonic, "Acceleration vs time (2)", x_label="Time [s]", y_label="Acceleration [m/s^2]")

n = 10
zero = 0
n_zero = 0
first_zero = 0
first_zero_time = 0
time_n = 0
end = 0
start = 0
for i in range(100, len(c_harmonic)):
    if equal(c_harmonic[i], 0, e=0.05):
        if n_zero:
            if not first_zero:
                start = i
                first_zero_time = harmonic_time[i]
                first_zero = 1
            zero += 1
            if zero == 19:
                time_n = harmonic_time[i] - first_zero_time
                end = i
                break
            n_zero = 0
    if equal(c_harmonic[i], 0.2, e=0.2):
        n_zero = 1
    if equal(c_harmonic[i], -0.2, e=0.2):
        n_zero = 1

T = time_n/n
print("Period: ", T)
print("Frequency (1): ", 1/T)

harmonic_fft = scipy.fft.fft(c_harmonic)[0:480]
f = scipy.fft.fftfreq(len(harmonic_time), harmonic_time[1]-harmonic_time[0])[0:480]
harmonic_fft_max = np.max(harmonic_fft)
f2 = 0
for i in range(len(harmonic_fft)):
    if harmonic_fft_max == harmonic_fft[i]:
        f2 = f[i]
        print("Frequency (2): ", f[i])

create_graph(f, np.abs(harmonic_fft), "Fourier Transform", x_label="Frequency [Hz]", y_label="Amplitude",
             x_ticks=[0, f2, 2, 4, 6, 8, 10])


def func(x, f):
    return 0.5 * np.cos(2*np.pi*f*x) * np.exp(-0.01*x)


def func2(x, f, a, b):
    return a * np.cos(2*np.pi*f*x) * np.exp(-b*x)


fit_x = [0.05*i for i in range(len(c_harmonic))]
fit_params, matrix = curve_fit(func2, fit_x, c_harmonic)

print("Frequency (3): ", fit_params[0])
print("Factor a: ", fit_params[1])
print("Factor b: ", fit_params[2])


fit_values = []
for i in range(len(harmonic_time)):
    fit_values.append(func(harmonic_time[i], fit_params[0]))

fit_values = []
for i in range(len(harmonic_time)):
    fit_values.append(func2(harmonic_time[i], fit_params[0], fit_params[1], fit_params[2]))

create_graph(fit_x, c_harmonic, "Acceleration vs time (3)", x_label="Time [s]", y_label="Acceleration [m/s^2]",
             continuous=True, label=["Dane pomiarowe", "Dopasowana funkcja"], values=[[fit_x, fit_values]])

# barometr
with open("data/pomiar2.txt", 'r') as f:
    data = f.read().split('\n')

i = []
pres = []
temp = []

for row in data:
    row_data = row.split("\t")
    i.append(int(row_data[0]) * 0.3)
    # i.append(int(row_data[0])/10)
    pres.append(round(float(row_data[1]), 4))
    temp.append(round(float(row_data[2]), 2))


temp_average = np.sum(temp)/len(temp) + 273.15
print("Average temperature [K]: ", temp_average)
p_ref = 1013.25  # hPa
R = 8.314
g = 9.81
u = 0.0289644

h = []
for p in pres:
    h.append(-np.log(p/p_ref)*R*temp_average/(u*g))

h0 = np.sum(h[350:400])/50
h1 = (np.sum(h[250:350]) + np.sum(h[450:500]))/150
h2 = (np.sum(h[150:200]) + np.sum(h[550:600]))/100
h3 = (np.sum(h[0:100]) + np.sum(h[630:700]))/170
h4 = (np.sum(h[710:760]) + np.sum(h[950:1000]))/100
h5 = np.sum(h[850:900])/50
hx = (h5, h4, h3, h2, h1, h0)

print(h0)
print(h1)
print(h2)
print(h3)
print(h4)
print(h5)

create_graph(i, pres, "Pressure vs time", x_label="Time [s]", y_label="Pressure [hPa]")
create_graph(i, h, "Height (time) ", x_label="Time [s]", y_label="Height above sea level [m]")

N = 300
dpi = 150
plt.plot(i, h)
plt.axhline(h5, color=mcolors.CSS4_COLORS["darkred"], label='Piętro 5')
plt.axhline(h4, color=mcolors.CSS4_COLORS["maroon"], label='Piętro 4')
plt.axhline(h3, color=mcolors.CSS4_COLORS["firebrick"], label='Piętro 3')
plt.axhline(h2, color=mcolors.CSS4_COLORS["brown"], label='Piętro 2')
plt.axhline(h1, color=mcolors.CSS4_COLORS["indianred"], label='Piętro 1')
plt.axhline(h0, color=mcolors.CSS4_COLORS["lightcoral"], label='Piętro 0')
plt.legend(loc="lower right")
plt.title("Height vs time")
plt.xlabel("Time [s]")
plt.ylabel("Height above sea level [m]")
plt.grid()
plt.savefig("Height vs time" + '.png', dpi=dpi)
plt.clf()
