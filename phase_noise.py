# Simulation on phase noise

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

fs = 5e6
f0 = 100e3
T = 1 / fs
t = np.arange(0, 1e-3, T)
N = len(t)

# Phase noise as integrated white noise (Wiener process)
np.random.seed(0)
phase_noise = np.cumsum(np.random.randn(len(t)) * 0.01)  # adjust 0.01 for severity

# Clean signal
signal_clean = np.cos(2 * np.pi * f0 * t)
# Noisy signal
signal_noisy = np.cos(2 * np.pi * f0 * t + phase_noise)

# FFT
x_clean = fft(signal_clean)
x_noisy = fft(signal_noisy)
x_clean_mag = np.abs(x_clean) / N
x_noisy_mag = np.abs(x_noisy) / N
x_freq = fftfreq(N, 1/fs)


fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].plot(t, signal_clean, color='green', label='clean signal')
ax[0].plot(t, signal_noisy, color='red', label='noisy signal')
ax[0].set_xlim(0, 1/f0*10)  # Limit x-axis to positive frequencies, 2xf0
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Time domain signals')
ax[0].legend()

ax[1].plot(x_freq/1e3, x_clean_mag, color='green',label='clean signal spectrum')
ax[1].plot(x_freq/1e3, x_noisy_mag, color='red', label='noisy signal spectrum')
ax[1].set_xlim(0, f0*2/1e3)  # Limit x-axis to positive frequencies, 2xf0
ax[1].set_xlabel('Frequency (kHz)')
ax[1].set_ylabel('Magnitude')
ax[1].set_title('Frequency domain spectrum showing phase noise')
ax[1].legend()
plt.show()
