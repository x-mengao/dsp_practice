# ##############################        
# Problem: Problem: You are given a reference signal and a noisy signal that contains the reference signal with some delay and noise. 
# Use cross-correlation to detect the presence of the reference signal in the noisy data.

# Cross-correlation is a powerful technique in signal processing used to detect the presence and 
# location of a known signal (reference) within a possibly noisy or complex received signal. 
# It works by comparing the received signal with the reference signal, finding the lag (time delay) 
# at which the maximum similarity occurs, indicating the location of the reference signal within the received signal. 

# Steps:
# 1. Generate a reference signal (e.g., a sine wave).
# 2. Generate a noisy signal by adding noise to the reference signal and introducing a delay.
# 3. Use cross-correlation to detect the presence of the reference signal in the noisy data.
#    The result of this sliding and multiplying process is a cross-correlation function. 
#    The peak in this function indicates the time delay (lag) where the two signals have the greatest similarity.
# ##############################  

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import find_peaks

def subplot_label_tool(subplot, label):
    subplot.set_title(label)
    subplot.set_xlabel('Time [s]')
    subplot.set_ylabel('Amplitude')

fs = 1000  # Sampling frequency
sig_freq = 50  # Frequency of the reference signal
num_samples = 500  # Number of samples
delay = 30 # Delay in samples
amplitude = 1

# Generate a reference signal (sine wave)
t = np.linspace(0, num_samples/fs, num_samples)
ref_sig = amplitude * np.sin(2 * np.pi * sig_freq * t)
noise = np.random.normal(0, 0.5, num_samples)  # Generate noise

noisy_sig = amplitude * np.sin(2 * np.pi * sig_freq * t + delay) + noise # Noisy signal with delay

# noisy_sig = amplitude * np.sin(2 * np.pi * sig_freq * t + delay) + noise # Noisy signal with delay and noisy 

# Running cross-correlation
z = correlate(noisy_sig, ref_sig, mode='full') # The output is the full discrete linear cross-correlation of the inputs. (Default)
# The length of the output is (N + M - 1), where N and M are the lengths of the input signals.

# plotting the signals
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(t, ref_sig, label='Reference Signal', color='blue')
subplot_label_tool(ax[0], 'Reference Signal')
ax[1].plot(t, noisy_sig, label='Noisy Signal', color='red')
subplot_label_tool(ax[1], 'Noisy Signal')
ax[2].plot(z, label='Cross-Correlation', color='green')
ax[2].set_title('Cross-Correlation')
ax[2].set_xlabel('Samples')
ax[2].set_ylabel('Amplitude')

# find maximum peak for max correlation
max_corr = np.max(z)
max_corr_index = np.argmax(z)
lag = max_corr_index - (len(ref_sig) - 1) # lag in samples
print(f'Maximum correlation: {max_corr}')
print(f'Lag (delay) in samples: {lag}')

plt.tight_layout()
plt.show()
