# Simulating 3rd order intermodulation (IMD)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert

# Simulation parameters
fs = 1e6  # Sampling frequency 1 MHz
t = np.arange(0, 2e-3, 1/fs)  # 2 ms time vector, in step 1/fs 

# Two-tone input signal
f1 = 100e3  # 100 kHz
f2 = 110e3  # 110 kHz
A1 = 1.0
A2 = 1.0
x = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t)

# Nonlinear amplifier models
def nonlinear_amplifier(x, linearity_level):
    """
    Simulates nonlinearity using a polynomial model.
    linearity_level controls how strong the nonlinearity is.
    """
    return x + linearity_level * x**3

# Create signals with low and high linearity (low and high distortion)
y_low_linearity = nonlinear_amplifier(x, 0.8)   # Strong nonlinearity
# y_high_linearity = nonlinear_amplifier(x, 0.05)  # Mild nonlinearity
y_high_linearity = nonlinear_amplifier(x, 0.01)  # Small nonlinearity


# Compute FFT
def compute_fft(signal):
    """
    Input: time-domain signal to be compuated for FFT
    Output: *Only positive region of FFT spectrum* frequency, log-scale fft magnitude normalized

    Note: fft_freqs[:N//2] is used to select only the positive frequency components of the FFT result for real-valued signals
    """
    N = len(signal)
    fft_vals = np.fft.fft(signal * np.hanning(N))
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    return fft_freqs[:N//2], 20 * np.log10(np.abs(fft_vals[:N//2]) / np.max(np.abs(fft_vals))) 

# Labeling tool for FFT
def fft_plot_labels(plt, plot_title):
    plt.title(plot_title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")

def fft_subplot_labels(ax, subplot_title):
    ax.set_title(subplot_title)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-200, 10)
    ax.grid(True)

# FFT of both outputs
freqs_low, magnitude_low = compute_fft(y_low_linearity)
freqs_high, magnitude_high = compute_fft(y_high_linearity)
magnitude_low = np.array(magnitude_low)
magnitude_high = np.array(magnitude_high)

# Expected beat tones based on f1 = 100kHz, f2 = 110kHz
# Freqs         |       Descriptions
# 90            |       2f1 - f2
# 100, 110      |       f1, f2 carrier tones
# 120           |       2f2 - f1
# 190-330       |       2nd/3rd harmonics
# >330          |       higher order mixing tones, 2f1 + f2, 2f2 + f1

# Frequencies to annotate
tones = {
    "f1 (100kHz)": 100e3,
    "f2 (110kHz)": 110e3,
    "2f1 - f2 (90kHz)": 90e3,
    "2f2 - f1 (120kHz)": 120e3,
    # "f1 + f2 (210kHz)": 210e3,
    # "2f1 (200kHz)": 200e3,
    # "2f2 (220kHz)": 220e3,
    "3f1 (300kHz)": 300e3,
    "3f2 (330kHz)": 330e3,
    "2f1 + f2 (310kHz)": 310e3,
    "2f2 + f1 (320kHz)": 320e3,
}

# Plot with annotations
fig, ax = plt.subplots(2, 1, figsize=(14,8))
ax[0].plot(freqs_low / 1e3, magnitude_low, label='"Spectrum with Low Linearity (Strong Nonlinearity)"')
ax[1].plot(freqs_low / 1e3, magnitude_high, label='"Spectrum with High Linearity (Weak Nonlinearity)"')

text_position = 50
text_side_sign = -1
for label, freq in tones.items():
    text_xdelta = text_position * text_side_sign
    text_ydelta = -text_position 
    idx = np.argmin(np.abs(freqs_low - freq))
    ax[0].annotate(label, xy=(freqs_low[idx] / 1e3, magnitude_low[idx]), xytext=(freqs_low[idx] / 1e3 + text_xdelta, magnitude_low[idx]),
                 arrowprops=dict(arrowstyle="->", color='red'), fontsize=8, color='red', ha='center')
    ax[1].annotate(label, xy=(freqs_low[idx] / 1e3, magnitude_high[idx]), xytext=(freqs_low[idx] / 1e3 + text_xdelta, magnitude_high[idx]),
                arrowprops=dict(arrowstyle="->", color='green'), fontsize=8, color='green', ha='center')
    text_side_sign = -text_side_sign
fft_subplot_labels(ax[0], "Spectrum with Low Linearity (Strong Nonlinearity)")
fft_subplot_labels(ax[1], "Spectrum with High Linearity (Weak Nonlinearity)")
plt.subplots_adjust(hspace=0.3)
plt.show()
