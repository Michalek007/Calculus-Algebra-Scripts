import numpy as np
import matplotlib.pyplot as plt

# circuit data
R = 1
L = 1
C = 0.25
E = 1
w = 3.5 ** 0.5
f = w / (2 * np.pi)

# impedance
Zc = -1j / (w*C)
Zl = 1j * w * L

I = E / (R + Zc + Zl)

# voltage across capacitor
Uc = I * Zc


def symbolic_method(x):
    return np.abs(x), np.angle(x, deg=True)


t = np.linspace(0, 10, 1000)

amp, phase = symbolic_method(Uc)
Ut = amp * np.sin(w*t + phase)
Qt = Ut * C

I_amp, I_phase = symbolic_method(I)
It = amp * np.sin(w*t + phase)

print(f'Circuit data: R={R}, L={L}, C={C}, w={"%.3f" % w}, f={"%.3f" % f}, E=sin({"%.3f" % w}*t)\n')

print(f'U(t): {"%.3f" % amp} * sin(t * {"%.3f" % w} {"%.3f" % phase})\n')

print(f'I(t): {"%.3f" % I_amp} * sin(t * {"%.3f" % w} + {"%.3f" % I_phase}) \n')

print(f'Q(t): {"%.3f" % (C * amp)} * sin(t * {"%.3f" % w} {"%.3f" % phase})\n')


plt.plot(t, Ut, label='Voltage')
plt.plot(t, It, label='Current')
plt.plot(t, Qt, label='Charge')
plt.title("Capacitor in RLC circuit")
plt.xlabel("Time [s] ")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
