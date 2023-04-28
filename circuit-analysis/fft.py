import numpy as np
import matplotlib.pyplot as plt
import scipy

# amplitude and frequency for all signals
amp = 1
freq = 1000

# low pass filter data
Rl = 1.5 * 10**3  # 1.5 kOhm
Cl = 47 * 10**-9  # 47 nF

# high pass filter data
Rh = 1.5 * 10**3  # 1.5 kOhm
Ch = 10 * 10**-9  # 47 nF

# time domain <0, 1>
sampling = 88_000
t_lim = 1
time = np.linspace(0, t_lim, sampling, endpoint=False)
N = len(time)

# signals values in time domain defined above
sine = np.sin(2 * np.pi * freq * time)

square25 = scipy.signal.square(2 * np.pi * freq * time, 0.25)

square50 = scipy.signal.square(2 * np.pi * freq * time, 0.5)

triangle50 = scipy.signal.sawtooth(2 * np.pi * freq * time, 0.5)

triangle40 = scipy.signal.sawtooth(2 * np.pi * freq * time, 0.4)


def fourier_transform(y, t):
    Nt = len(t)
    dt = t[1] - t[0]
    yf = np.abs(scipy.fft.fft(y)[0:Nt // 2])
    for i in range(len(yf)-1):
        if yf[i] < 0.1:
            yf[i] = 0.0001
    phase = np.angle(scipy.fft.fft(y)[0:Nt // 2], deg=True)
    xf = np.fft.fftfreq(Nt, d=dt)[0:Nt // 2]
    return xf, yf, phase


def low_pass_filter_RC(y, t):
    return scipy.signal.convolve(y, 1 / (Rl * Cl) * np.exp(-t / (Rl * Cl)))[0:N]


def high_pass_filter_RC(y, t):
    # return y - scipy.signal.convolve(y,  1 / (R * C) * np.exp(-t / (R * C)))[0:N]
    Nt = len(t)
    dt = t[1] - t[0]
    f_y = np.linspace(0, t_lim, sampling)
    a = Rh * Ch / (Rh*Ch + dt)
    f_y[0] = y[0]
    for i in range(2, Nt):
        f_y[i] = a * (f_y[i-1] + a * (y[i] - y[i-1]))
    return f_y


def display_fourier_graph(x, y, phase, A1, title):
    # phase graph
    plt.grid()
    plt.xlim([0, 10000])
    plt.ylim([-180, 180])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [degree] ")
    plt.title(f"Phase: {title}")
    plt.plot(x, phase)
    plt.show()

    # log scale with A1 as reference signal
    y = 20 * np.log10(y / A1)

    # amplitude graph
    plt.grid()
    plt.xlim([0, 10000])
    plt.ylim([-40, 0])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB] ")
    plt.title(title)
    plt.plot(x, y)
    plt.show()


# signals values in time domain filtered in low pass filter
lp_filtered_square50 = low_pass_filter_RC(square50, time)
lp_filtered_square50 = lp_filtered_square50/np.max(lp_filtered_square50)

lp_filtered_triangle40 = low_pass_filter_RC(triangle40, time)
lp_filtered_triangle40 = lp_filtered_triangle40/np.max(lp_filtered_triangle40)

# signals values in time domain filtered in high pass filter
hp_filtered_square50 = high_pass_filter_RC(square50, time)
hp_filtered_square50 = hp_filtered_square50/np.max(hp_filtered_square50)

hp_filtered_triangle40 = high_pass_filter_RC(triangle40, time)
hp_filtered_triangle40 = hp_filtered_triangle40/np.max(hp_filtered_triangle40)

signals = {"Sine": sine,
           "Rectangular, duty:25%": square25,
           "Rectangular, duty:50%": square50,
           "Triangular, symmetry factor:50%": triangle50,
           "Low pass filtered triangular, symmetry factor:40%": lp_filtered_triangle40,
           "Low pass filtered rectangular, duty:50%": lp_filtered_square50,
           "High pass filtered triangular, symmetry factor:40%": hp_filtered_triangle40,
           "High pass filtered rectangular, duty:50%": hp_filtered_square50,
           }

A1_rectangular50 = 0

for title, signal in signals.items():
    xf, yf, phase = fourier_transform(signal, time)
    A1 = np.max(yf)
    filtered_factor = 0.1
    for i in range(len(phase)-1):
        if yf[i] < A1 * filtered_factor:
            phase[i] = 0
    if title == "Rectangular, duty:50%":
        A1_rectangular50 = A1
    if title == "High pass filtered rectangular, duty:50%" or title == "Low pass filtered rectangular, duty:50%":
        A1 = A1_rectangular50
    if title == "Low pass filtered triangular, symmetry factor:40%" or title == "High pass filtered triangular, symmetry factor:40%":
        A1 = np.max(fourier_transform(triangle40, time)[1])
    display_fourier_graph(xf, yf, phase, A1, title)
