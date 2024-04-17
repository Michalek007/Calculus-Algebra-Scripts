import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from pyplot_api import Graph

df2 = pd.read_csv('data/amplifier.csv')

# measured power [dBm]
pin = df2['Pin']
pout_1G = df2['Pout_1G']
pout_2G = df2['Pout_2G']

# gain of amplifier [dB]
gain_1G = pout_1G - pin
gain_2G = pout_2G - pin

#  gain (f=1GHz) = 15 dB
#  compensation point = gain -1dB = 14dB
diff_14dB_list = list(map(lambda arg: 14-arg, gain_1G))
diff_14dB_list = np.abs(diff_14dB_list)
CP1dB_1GHz_index = np.argmin(diff_14dB_list)
CP1dB_1GHz = (pin[CP1dB_1GHz_index], pout_1G[CP1dB_1GHz_index])
print('CP-1dB (f=1GHz):')
print('Pin:', CP1dB_1GHz[0], 'dBm')
print('Pout:', CP1dB_1GHz[1], 'dBm')
print('')

#  gain (f=2GHz) = 16 dB
#  compensation point = gain -1dB = 15dB
diff_15dB_list = list(map(lambda arg: 15-arg, gain_2G))
diff_15dB_list = np.abs(diff_15dB_list)
CP1dB_2GHz_index = np.argmin(diff_15dB_list)
CP1dB_2GHz = (pin[CP1dB_2GHz_index], pout_2G[CP1dB_2GHz_index])
print('CP-1dB (f=2GHz):')
print('Pin:', CP1dB_2GHz[0], 'dBm')
print('Pout:', CP1dB_2GHz[1], 'dBm')
print('')

# # # power in of power out
plt.plot(CP1dB_1GHz[0], CP1dB_1GHz[1], 'x', color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB_1GHz=({CP1dB_1GHz[0]}, {CP1dB_1GHz[1]})[dBm]')
plt.plot(CP1dB_2GHz[0], CP1dB_2GHz[1], 'x', color=mcolors.CSS4_COLORS['red'], label=f'CP-1dB_2GHz=({CP1dB_2GHz[0]}, {CP1dB_2GHz[1]})[dBm]')

# plt.axvline(CP1dB_1GHz[0], color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB_1GHz={CP1dB_1GHz[0]}dBm')
# plt.axvline(CP1dB_2GHz[0], color=mcolors.CSS4_COLORS['orange'], label=f'CP-1dB_2GHz={CP1dB_2GHz[0]}dBm')

# plt.axhline(CP1dB_1GHz[1], color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB_1GHz={CP1dB_1GHz[1]}dBm')
# plt.axhline(CP1dB_2GHz[1], color=mcolors.CSS4_COLORS['orange'], label=f'CP-1dB_2GHz={CP1dB_2GHz[1]}dBm')

graph = Graph(mode=True, label=['f=1GHz', 'f=2GHz'], x_label='Pin [dBm]', y_label='Pout [dBm]', loc='upper left')
graph.create_graph(pin, pout_1G, title='Charakterystyki przejściowe', values=[[pin, pout_2G]])

# # # gain of power in
graph = Graph(mode=True, label=['f=1GHz', 'f=2GHz'], x_label='Pin [dBm]', y_label='Gain [dB]')
graph.create_graph(pin, gain_1G, title='Wzmocnienie od mocy wejściowej', values=[[pin, gain_2G]])

# # # power in of power out (f=1GHz)
graph = Graph(mode=True, label=['Real amp', 'Ideal amp', 'Slope 3dB/dB'], x_label='Pin [dBm]', y_label='Pout [dBm]', loc='upper left')


fit_params, matrix = curve_fit(lambda x, a, b: a * x + b, pin[0:19], pout_1G[0:19])

step = 0.01
x = np.arange(-30, 10, step)
y_ideal = [x[i] * fit_params[0] + fit_params[1] for i in range(len(x))]
y_3dB = [3*x[i] for i in range(len(x))]

# common point is between 5 and 10 dBm
steps = 40 / step
start_index = int(35/40 * steps)
diff_IP3_list = []
for i in range(start_index, int(steps)): # -300, -150
    diff_IP3_list.append(y_ideal[i]-y_3dB[i])


diff_IP3_list = np.abs(diff_IP3_list)
IP3_index = np.argmin(diff_IP3_list)

IP3_index = IP3_index + start_index
IIP3, OIP3 = x[IP3_index], y_ideal[IP3_index]

print(f'Common point: y_ideal: ({round(x[IP3_index], 2)}, {round(y_ideal[IP3_index], 2)}), y_3dB: ({round(x[IP3_index], 2)}, {round(y_3dB[IP3_index], 2)})')
print('')

print('IP3 (f=1GHz):')
print('IIP3:', round(x[IP3_index], 2), 'dBm')
print('OIP3:', round(y_ideal[IP3_index], 2), 'dBm')

plt.plot(IIP3, OIP3, 'x', color=mcolors.CSS4_COLORS['red'], label=f'IP3=({round(IIP3, 2)}, {round(OIP3, 2)})[dBm]')
plt.plot(CP1dB_1GHz[0], CP1dB_1GHz[1], 'x', color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB=({CP1dB_1GHz[0]}, {CP1dB_1GHz[1]})[dBm]')

# plt.axvline(CP1dB_1GHz[0], color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB={CP1dB_1GHz[0]}dBm')
# plt.axhline(CP1dB_1GHz[1], color=mcolors.CSS4_COLORS['blue'], label=f'CP-1dB={CP1dB_1GHz[1]}dBm')

graph.create_graph(pin, pout_1G, title='Charakterystyka przejściowa, f=1GHz',
                   values=[
                       [x, y_ideal],
                       [x, y_3dB]
                   ])
