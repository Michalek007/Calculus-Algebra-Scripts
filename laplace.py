import scipy
import numpy as np
import matplotlib.pyplot as plt


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, r_signal=None):
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
        plt.plot(x, signal_in, label='Input')
        plt.legend(loc="upper left")
    plt.show()


def convolve(x1, x2, n):
    result = scipy.signal.convolve(x1, x2)[0:n]
    return result / np.max(result)


# amplitude, frequency and offset for input signal
freq = 100
amp = 1
offset = 1

# time domain <0, 1>
sampling = 88_000
t_lim = 0.2
time = np.linspace(0, t_lim, sampling, endpoint=False)
N = len(time)

# input signal - square with 50% duty cycle
signal_in = (scipy.signal.square(2 * np.pi * freq * time, 0.5) + offset)/2

display_graph(time, signal_in, "Input signal", [0, 0.05])


def low_pass_filter_RC(y, t):
    # low pass filter data
    R = 1.5 * 10 ** 3  # 1.5 kOhm
    C = 47 * 10 ** -9  # 47 nF
    return convolve(y, 1 / (R * C) * np.exp(-t / (R * C)), N)


def high_pass_filter_RC(y, t):
    # high pass filter data
    R = 1.5 * 10 ** 3  # 1.5 kOhm
    C = 10 * 10 ** -9  # 10 nF
    dt = t[1] - t[0]
    f_y = np.linspace(0, t_lim, sampling)
    a = R * C / (R*C + dt)
    f_y[0] = y[0]
    for i in range(2, N):
        f_y[i] = a * (f_y[i-1] + a * (y[i] - y[i-1]))
    return f_y * 1.3
    # return R * C * y - convolve(y,  - R * C * np.exp(-t / (R * C)), N)


def chebyshev_filter(y, t):
    L1 = 680 * 10**-6  # 680 uH
    C2 = 1.33 * 10**-6  # 1.33 uF
    a1 = ((L1 * C2) ** 0.5)
    sin = np.sin(time * a1)
    return convolve(y, - a1 * sin, N)


def RLC(t):
    # RLC circuit with 3 different configuration (different resistors)
    L = 0.01  # 10 mH
    C = 33 * 10**-9  # 33 nF
    resistance = (55, 1155, 3355)  # 55, 1155, 3355 Ohm
    voltage = []
    for R in resistance:
        a1 = (1 / (L * C) - (R ** 2) / (4 * L ** 2))
        if a1 > 0:
            a2 = np.sqrt(a1)
        else:
            a2 = np.abs(a1 ** 0.5)
        a3 = 2 / C - (R ** 2) / (2 * L)
        a4 = R * a2 / a3
        e = np.exp(-(R * t) / (2 * L))
        cos = np.cos(t * a2)
        sin = np.sin(t * a2)
        voltage.append(e * (cos - a4 * sin))
    return voltage


low_pass_output = low_pass_filter_RC(signal_in, time)
high_pass_output = high_pass_filter_RC(signal_in, time)
output_signal = RLC(time)

display_graph(time, low_pass_output, "Low pass filter", [0.0098, 0.0106], r_signal=signal_in)
display_graph(time, high_pass_output, "High pass filter", [0.00995, 0.0101], [-0.05, 1.05], r_signal=signal_in)
display_graph(time, output_signal[0], "Drgania tłumione RLC", [0, 0.002])
display_graph(time, output_signal[1], "Przebieg aperiodyczny - krytyczny RLC", [0, 0.0002])
display_graph(time, output_signal[2], "Przebieg aperiodyczny RLC", [0, 0.0002])


lp_tau = 70 * 10**-6  # 70us
hp_tau = 15 * 10**-6  # 15us

# voltage after tau, 5*tau, 10*tau
lp_tau_values = []
hp_tau_values = []

for i in range(N-1):
    if lp_tau + 0.000001 >= time[i] >= lp_tau - 0.000001:
        lp_tau_values.append(low_pass_output[i])
    if hp_tau + 0.000001 >= time[i] >= hp_tau - 0.000001:
        hp_tau_values.append(high_pass_output[i])
    if 5*lp_tau + 0.000001 >= time[i] >= 5*lp_tau - 0.000001:
        lp_tau_values.append(low_pass_output[i])
    if 5*hp_tau + 0.000001 >= time[i] >= 5*hp_tau - 0.000001:
        hp_tau_values.append(high_pass_output[i])
    if 10*lp_tau + 0.000001 >= time[i] >= 10*lp_tau - 0.000001:
        lp_tau_values.append(low_pass_output[i])
    if 10*hp_tau + 0.000001 >= time[i] >= 10*hp_tau - 0.000001:
        hp_tau_values.append(high_pass_output[i])
    if time[i] > 0.01:
        break

lp_rise10 = 0
lp_rise90 = 0
hp_rise10 = 0
hp_rise90 = 0

for i in range(N-1):
    if 0.1 + 0.01 >= low_pass_output[i] >= 0.1 - 0.01:
        lp_rise10 = time[i]
    if 0.1 + 0.05 >= high_pass_output[i] >= 0.1 - 0.05:
        hp_rise10 = time[i]
    if 0.9 + 0.01 >= low_pass_output[i] >= 0.9 - 0.01:
        lp_rise90 = time[i]
    if 0.9 + 0.05 >= high_pass_output[i] >= 0.9 - 0.05:
        hp_rise90 = time[i]

# rise time
lp_rise_time = (lp_rise10 - lp_rise90) * 10**6
hp_rise_time = (hp_rise10 - hp_rise90) * 10**6

min_aperiodic = np.min(output_signal[2])
min_aperiodic_time = 0
for i in range(N-1):
    if output_signal[2][i] == min_aperiodic:
        min_aperiodic_time = time[i] * 10**6

maxima = scipy.signal.argrelmax(output_signal[0])
maxima_distance = (time[maxima[0][2]] - time[maxima[0][1]]) * 10**6

print(f"Minimum of aperiodic course: {min_aperiodic_time} us")
print(f"Distance between maximum for RLC oscillator: {maxima_distance} us")

for value in lp_tau_values:
    print(f"LP filter value after n * tau: {value} V")

for value in hp_tau_values:
    print(f"HP filter value after n * tau: {value} V")

print(f"LP filter rise time: {lp_rise_time} us")
print(f"HP filter rise time: {hp_rise_time} us")
