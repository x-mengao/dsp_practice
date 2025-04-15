# ##############################        
# Problem: You are provided with a set of discrete samples of a signal (e.g., sampled at 1 kHz), 
# and your task is to reconstruct the continuous signal using interpolation.
# Steps:
# 1. Generate a set of discrete samples of a sine wave signal.
# 2. Use linear interpolation and spline interpolation to reconstruct the continuous signal.
# 3. Plot the original signal and the reconstructed signals.
# ##############################  

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, CubicSpline

def subplot_label_tool(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Time [s]')
    subplot.set_ylabel('Amplitude')

# Step 1: Generate discrete samples of a sine wave signal
fs = 2000  # Sampling frequency (Hz)
num_samples = 50  # Number of samples
t = np.linspace(0, num_samples / fs, num_samples)

sig_freq = 123  # Frequency of the sine wave (Hz)
amplitude = 1.0  # Amplitude of the sine wave
sig = amplitude * np.sin(2 * np.pi * sig_freq * t)

# Step 2: Use linear interpolation and spline interpolation to reconstruct the continuous signal
# Linear interpolation
linear_interp = interp1d(t, sig, kind='linear')
# Spline interpolation
spline_interp = CubicSpline(t, sig)

# Step 3. Plotting the original and reconstructed signals
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(t, sig, 'o', label='Discrete samples of a sinewave', markersize=5)
ax[0].plot(t, linear_interp(t), '-', color='blue', label='Linear Interpolation', linewidth=2)
subplot_label_tool(ax[0], 'Linear interpolation')
ax[0].legend()

ax[1].plot(t, sig, 'o', label='Discrete samples of a sinewave', markersize=5)
ax[1].plot(t, spline_interp(t), '-', color='green', label='Spline Interpolation', linewidth=2)
subplot_label_tool(ax[1], 'Spline interpolation')
ax[1].legend()

plt.tight_layout()
plt.show()
