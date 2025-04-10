# ##############################        
# Problem: You have a time-domain signal x(t) which is the sum of three sine waves with 
# frequencies 50 Hz, 150 Hz, and 300 Hz. You want to analyze the frequency content of this signal.
# Steps
# 1. Generate the synthetic time-domain signal x(t).
# 2. Compute the FFT of the signal.
# 3. Plot the magnitude of FFT
# 4. Identify the peaks in spetrum
# ##############################        

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks

f1 = 50
f2 = 150
f3 = 300
a1 = 1
a2 = 0.5
a3 = 0.2
fs = 1000  # Sampling frequency
num_samples = 1000  # Number of samples
t = np.arange(num_samples) / fs  # Time vector

x = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t) + a3 * np.sin(2 * np.pi * f3 * t)
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(t, x)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Time Domain Synthetic Signal')

# Compute FFT of the signal
N = len(x)
X = fft(x)
X_magnitude = np.abs(X) / N  # Normalize the magnitude
frequencies = fftfreq(N, 1/fs)
# Plot the magnitude of FFT
ax[1].plot(frequencies, X_magnitude) 
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Magnitude')
ax[1].set_title('FFT Magnitude Spectrum')
ax[1].set_xlim(0, fs/2)  # Limit x-axis to positive frequencies

# Note: following are not necessary as fftfreq has already shifted the zero frequency component to the center
# frequencies = fftshift(frequencies)  # Shift zero frequency component to center, so it ranges from -fs/2 to fs/2-1bin
# ax[1].plot(frequencies, fftshift(X_magnitude)) # Shift FFT output so it aligns with shifted frequencies axis

# Identify peaks in the spectrum
peaks, _ = find_peaks(X_magnitude, height=0.1)  
ax[1].plot(frequencies[peaks],X_magnitude[peaks], "x")  
plt.tight_layout()
plt.show()