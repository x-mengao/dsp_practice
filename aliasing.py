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
    subplot.set_xlabel('Time [ms]')
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
    peak = np.argmax(X_magnitude)
    return frequencies, X_magnitude, peak

# Signal parameters and sampling frequency options
freq_sig = 500
fs1 = 600  # Sampling frequency 1
fs2 = 800  # Sampling frequency 2
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
fig, ax = plt.subplots(4, 2, figsize=(10, 8))
ax[0,0].plot(t1, sig_sampled1, label=f'fs={fs1}Hz')
ax[0,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[0,0], f'Sampling at {fs1}Hz, showing aliasing effect')
ax[1,0].plot(t2, sig_sampled2, label=f'fs={fs2}Hz')
ax[1,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[1,0], f'Sampling at {fs2}Hz, showing aliasing effect')
ax[2,0].plot(t3, sig_sampled3, label=f'fs={fs3}Hz')
ax[2,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[2,0], f'Sampling at {fs3}Hz, no aliasing')
ax[3,0].plot(t4, sig_sampled4, label=f'fs={fs4}Hz')
ax[3,0].set_xlim(0, 1/freq_sig * 10)
subplot_label_tool_time(ax[3,0], f'Sampling at {fs4}Hz, no aliasing')


# FFT analysis
freq1, mag1, peak1 = run_fft(sig_sampled1, fs1)
freq2, mag2, peak2 = run_fft(sig_sampled2, fs2)
freq3, mag3, peak3 = run_fft(sig_sampled3, fs3)
freq4, mag4, peak4 = run_fft(sig_sampled4, fs4)
print(f"Peak1: {freq1[peak1]}Hz, Peak2: {freq2[peak2]}Hz, Peak3: {freq3[peak3]}Hz, Peak4: {freq4[peak4]}Hz")

# Plotting FFT results
ax[0,1].plot(freq1, mag1, label=f'Spetrum of signal sampled at {fs1}Hz')
ax[0,1].plot(freq1[peak1], mag1[peak1], 'rx')
ax[0,1].set_xlim(0, freq_sig * 5)
subplot_label_tool_freq(ax[0,1], 'FFT of sampled signals')

ax[1,1].plot(freq2, mag2, label=f'Spectrum of signal sampled at {fs2}Hz')
ax[1,1].plot(freq2[peak2], mag2[peak2], 'rx')
ax[1,1].set_xlim(0, freq_sig * 5)
subplot_label_tool_freq(ax[1,1], 'FFT of sampled signals')

ax[2,1].plot(freq3, mag3, label=f'Spectrum of signal sampled at {fs3}Hz')
ax[2,1].plot(freq3[peak3], mag3[peak3], 'rx')
ax[2,1].set_xlim(0, freq_sig * 5)
subplot_label_tool_freq(ax[2,1], 'FFT of sampled signals')

ax[3,1].plot(freq4, mag4, label=f'Spectrum of signal sampled at {fs4}Hz')
ax[3,1].plot(freq4[peak4], mag4[peak4], 'rx')
ax[3,1].set_xlim(0, freq_sig * 5)
subplot_label_tool_freq(ax[3,1], 'FFT of sampled signals')

plt.tight_layout()
plt.show()  # Display the plots


