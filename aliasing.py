# ##############################        
# Problem: You are given a continuous signal of frequency 500 Hz, and you are asked to 
# sample this signal at different rates: 600 Hz, 1000 Hz, and 2000 Hz. 
# The Nyquist theorem suggests that sampling should occur at twice the highest frequency of the signal.

# Show the effect of aliasing when sampling below the Nyquist rate (i.e., at 600 Hz).
# ##############################        

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def subplot_label_tool_time(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Time [s]')
    subplot.set_ylabel('Amplitude')

def subplot_label_tool_freq(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Frequency [Hz]')
    subplot.set_ylabel('Magnitude')

def run_fft(signal, fs):
    N = len(signal)
    X = fft(signal)
    X_magnitude = np.abs(X) / N  # Normalize the magnitude
    frequencies = fftfreq(N, 1/fs)
    return frequencies, X_magnitude

freq_sig = 500
fs1 = 600  # Sampling frequency 1
fs2 = 1000  # Sampling frequency 2
fs3 = 2000  # Sampling frequency 3
fs4 = freq_sig * 25  # Sampling frequency 4, 10 times the signal frequency
nyqust_rate = 2 * freq_sig
num_samples = 500

t1 = np.arange(num_samples) / fs1
t2 = np.arange(num_samples) / fs2
t3 = np.arange(num_samples) / fs3
t4 = np.arange(num_samples) / fs4

# Sampled signals construction
sig_sampled1 = np.sin(2 * np.pi * freq_sig * t1)
sig_sampled2 = np.sin(2 * np.pi * freq_sig * t2)
sig_sampled3 = np.sin(2 * np.pi * freq_sig * t3)
sig_sampled4 = np.sin(2 * np.pi * freq_sig * t4)

# Plotting differences
fig, ax = plt.subplots(4, 2, figsize=(12, 10))
ax[0,0].plot(t1, sig_sampled1, label='fs=600 Hz')
ax[0,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[0,0], 'Sampling at 600Hz, showing aliasing effect')
ax[1,0].plot(t2, sig_sampled2, label='fs=1000 Hz')
ax[1,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[1,0], 'Sampling at 1000Hz, bad aliasing due to exact multiple of signal freq')
ax[2,0].plot(t3, sig_sampled3, label='fs=2000 Hz')
ax[2,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[2,0], 'Sampling at 2000Hz, no aliasing')
ax[3,0].plot(t4, sig_sampled4, label='fs=5000 Hz')
ax[3,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[3,0], 'Sampling at 5000Hz, no aliasing')


# FFT analysis
freq1, mag1 = run_fft(sig_sampled1, fs1)
freq2, mag2 = run_fft(sig_sampled2, fs2)
freq3, mag3 = run_fft(sig_sampled3, fs3)
freq4, mag4 = run_fft(sig_sampled4, fs4)
ax[0,1].plot(freq1, mag1, label='Spetrum of signal sampled at 600Hz')
ax[0,1].set_xlim(0, 2000)
subplot_label_tool_freq(ax[0,1], 'FFT of sampled signals')

ax[1,1].plot(freq2, mag2, label='Spectrum of signal sampled at 1000Hz')
ax[1,1].set_xlim(0, 2000)
subplot_label_tool_freq(ax[1,1], 'FFT of sampled signals')

ax[2,1].plot(freq3, mag3, label='Spectrum of signal sampled at 2000Hz')
ax[2,1].set_xlim(0, 2000)
subplot_label_tool_freq(ax[2,1], 'FFT of sampled signals')

ax[3,1].plot(freq4, mag4, label='Spectrum of signal sampled at 5000Hz')
ax[3,1].set_xlim(0, 2000)
subplot_label_tool_freq(ax[3,1], 'FFT of sampled signals')

plt.tight_layout()
plt.show()  # Display the plots


