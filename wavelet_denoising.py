# ############################## 
# Problem: You are given a signal that contains both noise and a low-frequency component. 
# The noise is high-frequency, and you need to recover the original signal while preserving the low-frequency information.
# Steps
# 1. Generate signal with noise
# 2. Decompose signal into Approximations(cA) and details(cD).
#    This process yields a set of approximation coefficients (low-frequency components) and detail coefficients 
#    (high-frequency components) at various levels.
# 3. Threshold the detail coefficients to remove high-frequency noise.
#    Coefficients below this threshold, assumed to represent noise, are either set to zero (hard thresholding) 
#    or reduced in magnitude (soft thresholding)
# 4. Reconstruct the signal from the modified coefficients.
# 5. Plot the original signal, noisy signal, and denoised signal.
# ############################## 

import numpy as np
import pywt
import matplotlib.pyplot as plt

def subplot_label_tool(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Time [s]')
    subplot.set_ylabel('Amplitude')

sample_rate = 1000  # Sampling rate in Hz
fc = 50  # Signal frequency at 50Hz
num_samples = 1000  # Number of samples

# Generate a time vector
t = np.arange(num_samples) / sample_rate
sig = np.sin(2 * np.pi * fc * t)  # Clean signal
noise = np.random.normal(0, 0.2, len(sig))
sig_noisy = sig + noise  # Noisy signal

# Decompose the noisy signal using wavelet
wavelet = 'db4'
coeffs = pywt.wavedec(sig_noisy, wavelet, level=2) # cA and cD1, cD2 are coefficients
cA, cD1, cD2 = coeffs # cA is approximation, cD1 and cD2 are details at 2 levels
# Note: pywt.wavedec is multi-level decomposition, pywt.dwt is single level only

# Thresholding
# https://dsp.stackexchange.com/questions/15464/wavelet-thresholding
# Hard thresholding makes |cD| < threshold to 0
# Soft thresholding makes |cD| < threshold to 0 and |cD| > threshold to |cD| - threshold

# Use universal threshold (Donoho-Johnstone)
# T = σ * sqrt(2 * log(n)) where σ is an estimate of the noise level 
# (typically the median absolute deviation of the coefficients) and n is the number of data points. 
threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(sig_noisy)))
coeffs_thresholded = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

# Reconstruct the signal
sig_denoised = pywt.waverec(coeffs_thresholded, wavelet)

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(t, sig)
subplot_label_tool(ax[0], 'Clean Signal')
ax[1].plot(t, sig_noisy, color='red')
subplot_label_tool(ax[1], 'Noisy Signal')
ax[2].plot(t, sig_denoised, color='green')
subplot_label_tool(ax[2], 'Denoised Signal')
plt.tight_layout()
plt.show()
