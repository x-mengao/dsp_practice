# ##############################        
# This is a challenge problem to design filter
# Problem: You are given a noisy signal that is sampled at 1 kHz. 
# The signal contains useful data at frequencies below 100 Hz and noise at frequencies above 200 Hz. 
# Design a low-pass filter to remove the high-frequency noise.
# ############################## 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def subplot_label_tool(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Time [s]')
    subplot.set_ylabel('Amplitude')

sample_rate = 1000  # Sampling rate in Hz
fc = 50  # Signal frequency at 50Hz
fn = 300  # Noise frequency at 300Hz
num_samples = 1000  # Number of samples

# Generate a time vector
t = np.arange(num_samples) / sample_rate
sig = np.sin(2 * np.pi * fc * t)  # Clean signal
sig_noisy = np.sin(2 * np.pi * fc * t) + np.sin(2 * np.pi * fn * t)  # Noisy signal
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(t, sig)
subplot_label_tool(ax[0], 'Clean Signal')
ax[1].plot(t, sig_noisy, color='red')
subplot_label_tool(ax[1], 'Noisy Signal')


# Designing the low-pass filter
b, a = butter(4, 100 / (sample_rate / 2), 'low')  # 4th order Butterworth filter
filtered_sig = lfilter(b, a, sig_noisy)
ax[2].plot(t, filtered_sig, color='orange')
subplot_label_tool(ax[2], 'Filtered Signal')

plt.tight_layout()
plt.show()
