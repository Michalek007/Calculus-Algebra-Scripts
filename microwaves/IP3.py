gain = 15  # gain for f=1GHz [dB]

# m1 & m2 -> podstawowe harmoniczne
# m3 & m4 -> zakłócenia intermodulacyjne trzeciego rzędu/

print('OIP3 IIP3')

# first image markers values [dBm], Pin=-15dBm
m1 = 11.4  # 1.1 GHz
m2 = 10.1  # 1 GHz
m3 = -0.1  # 1.2 GHz
m4 = -1.9  # 0.9 GHz

Pout = (m1 + m2) / 2
Pout_3 = (m3 + m4) / 2
delta_P = Pout - Pout_3
OIP3 = Pout + delta_P
IIP3 = OIP3 - gain
print(OIP3, IIP3)


# second image markers values [dBm], Pin=-10dBm
m1 = 10.4  # 1.1 GHz
m2 = 9  # 1 GHz
m3 = -9.5  # 1.2 GHz
m4 = -10.1  # 0.9 GHz

Pout = (m1 + m2) / 2
Pout_3 = (m3 + m4) / 2
delta_P = Pout - Pout_3
OIP3 = Pout + delta_P
IIP3 = OIP3 - gain
print(OIP3, IIP3)


# third image markers values [dBm], Pin=-5dBm
m1 = 7  # 1.1 GHz
m2 = 5.7  # 1 GHz
m3 = -28.9  # 1.2 GHz
m4 = -29.8  # 0.9 GHz

Pout = (m1 + m2) / 2
Pout_3 = (m3 + m4) / 2
delta_P = Pout - Pout_3
OIP3 = Pout + delta_P
IIP3 = OIP3 - gain
print(round(OIP3, 2), round(IIP3, 2))

# IP3 calculated from graphs
# IP3 (f=1GHz):
# IIP3: 7.25 dBm
# OIP3: 21.76 dBm
