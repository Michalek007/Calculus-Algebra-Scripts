from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


def display_graph(x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = "Frequency [Hz]",
                  y_label: str = "Amplitude [dB]", continuous=True, r_signal=None):
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


lp_hp_filter_data = genfromtxt('filters_data\\LowHighPassFilter.csv', skip_header=True, delimiter=',')
chebyshev_data = genfromtxt('filters_data\\ChebyshevFilter.csv', skip_header=True, delimiter=',')
rlc_jp41_data = genfromtxt('filters_data\\RLC_JP41.csv', skip_header=True, delimiter=',')
rlc_jp42_jp43_data = genfromtxt('filters_data\\RLC_JP42_JP43.csv', skip_header=True, delimiter=',')

Np = len(lp_hp_filter_data)
freq_lp_hp = np.zeros(Np)
lp_uwe = np.zeros(Np)
lp_uwy = np.zeros(Np)
lp_phase = np.zeros(Np)
lp_k = np.zeros(Np)
hp_uwe = np.zeros(Np)
hp_uwy = np.zeros(Np)
hp_phase = np.zeros(Np)
hp_k = np.zeros(Np)

i = 0
for row in lp_hp_filter_data:
    freq_lp_hp[i] = row[0]
    lp_uwe[i] = row[1]
    lp_uwy[i] = row[2]
    lp_phase[i] = row[3]
    lp_k[i] = row[2]/row[1]
    hp_uwe[i] = row[4]
    hp_uwy[i] = row[5]
    hp_phase[i] = row[6]
    hp_k[i] = row[5]/row[4]
    i += 1
lp_k = 20 * np.log10(lp_k)
hp_k = 20 * np.log10(hp_k)

display_graph(freq_lp_hp, lp_k, "Filtr dolnoprzepustowy - charakterystyka amplitudowa")
display_graph(freq_lp_hp, lp_phase, "Filtr dolnoprzepustowy - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(lp_phase, lp_k, "Filtr dolnoprzepustowy - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]")

display_graph(freq_lp_hp, hp_k, "Filtr górnoprzepustowy - charakterystyka amplitudowa")
display_graph(freq_lp_hp, hp_phase, "Filtr górnoprzepustowy - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(hp_phase, hp_k, "Filtr górnoprzepustowy - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]")

Nc = len(chebyshev_data)
freq_chebyshev = np.zeros(Nc)
c_uwe = np.zeros(Nc)
c_uwy = np.zeros(Nc)
c_phase = np.zeros(Nc)
c_k = np.zeros(Nc)

i = 0
for row in chebyshev_data:
    freq_chebyshev[i] = row[0]
    c_uwe[i] = row[1]
    c_uwy[i] = row[2]
    c_phase[i] = row[3]
    c_k[i] = row[2]/row[1]
    i += 1
c_k = 20 * np.log10(c_k)

display_graph(freq_chebyshev, c_k, "Filtr Czebyszewa - charakterystyka amplitudowa ")
display_graph(freq_chebyshev, c_phase, "Filtr Czebyszewa - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(c_phase, c_k, "Filtr Czebyszewa - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]")


N_rlc1 = len(rlc_jp41_data)
freq_rlc1 = np.zeros(N_rlc1)
jp41_uwe = np.zeros(N_rlc1)
jp41_uwy = np.zeros(N_rlc1)
jp41_phase = np.zeros(N_rlc1)
jp41_k = np.zeros(N_rlc1)

i = 0
for row in rlc_jp41_data:
    freq_rlc1[i] = row[0]
    jp41_uwe[i] = row[1]
    jp41_uwy[i] = row[2]
    jp41_phase[i] = row[3]
    jp41_k[i] = row[2]/row[1]
    i += 1

jp41_k = 20 * np.log10(jp41_k)

display_graph(freq_rlc1, jp41_k, "Układ RLC z rezystancją 50 Ω - charakterystyka amplitudowa")
display_graph(freq_rlc1, jp41_phase, "Układ RLC z rezystancją 50 Ω - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(jp41_phase, jp41_k, "Układ RLC z rezystancją 50 Ω - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]")


N_rlc = len(rlc_jp42_jp43_data)
freq_rlc = np.zeros(N_rlc)
jp42_uwe = np.zeros(N_rlc)
jp42_uwy = np.zeros(N_rlc)
jp42_phase = np.zeros(N_rlc)
jp42_k = np.zeros(N_rlc)
jp43_uwe = np.zeros(N_rlc)
jp43_uwy = np.zeros(N_rlc)
jp43_phase = np.zeros(N_rlc)
jp43_k = np.zeros(N_rlc)

i = 0
for row in rlc_jp42_jp43_data:
    freq_rlc[i] = row[0]
    jp42_uwe[i] = row[1]
    jp42_uwy[i] = row[2]
    jp42_phase[i] = row[3]
    jp42_k[i] = row[2]/row[1]
    jp43_uwe[i] = row[4]
    jp43_uwy[i] = row[5]
    jp43_phase[i] = row[6]
    jp43_k[i] = row[5]/row[4]
    i += 1
jp43_k = 20 * np.log10(jp43_k)
jp42_k = 20 * np.log10(jp42_k)

display_graph(freq_rlc, jp42_k, "Układ RLC z rezystancją 1.15 kΩ - charakterystyka amplitudowa")
display_graph(freq_rlc, jp42_phase, "Układ RLC z rezystancją 1.15 kΩ - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(jp42_phase, jp42_k, "Układ RLC z rezystancją 1.15 kΩ - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]")

display_graph(freq_rlc, jp43_k, "Układ RLC z rezystancją 3.35 kΩ - charakterystyka amplitudowa")
display_graph(freq_rlc, jp43_phase, "Układ RLC z rezystancją 3.35 kΩ - charakterystyka fazowa", y_label="Phase [degree]")
display_graph(jp43_phase, jp43_k, "Układ RLC z rezystancją 3.35 kΩ - charakterystyka amplitudowo-fazowa",
              x_label="Phase [degree]", y_label="Amplitude [dB]", continuous=False)
