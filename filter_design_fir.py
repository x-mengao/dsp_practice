# ################################
# Design a FIR filter with following specs:
# Low pass FIR filter, 1kHz cutoff, 40dB stopband attanuation at 1.5kHz
# Use scipy library, do plot to show freq response

# Design method used here: windowing
# ################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Specs for filter
fs = 8000
f_cutoff = 1000
trans_width = 500 # stopband 1500 - cutoff 1000
nyquist_freq = fs/2

# Estimate number of taps using RULE OF THUMB
# N ~= 4 / (transition_width / fs)
num_taps = int(np.ceil(4 / (trans_width / fs)))

# use firwin to design filter
taps = firwin(num_taps, cutoff=f_cutoff, window = 'hamming', fs=fs)

# Freq response
w, h = freqz(taps, worN=8000, fs=fs) 
# worN usage: If a single integer, then compute at that many frequencies (default is N=512). 
# This is a convenient alternative to:
# np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)

# plot
plt.figure(figsize = (10, 6))
plt.plot(w, 20*np.log10(np.abs(h)), label='Frequency response')
plt.axvline(f_cutoff, color='green', linestyle='--', label='Cutoff = 1kHz')
plt.axvline(f_cutoff + trans_width, color='red', linestyle='--', label='Stopband starts = 1.5kHz')
plt.title('FIR Low-pass Filter using Hamming Window')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.ylim(-80, 5)
plt.grid(True)
plt.legend()
plt.show()
