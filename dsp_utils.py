# This script hosts common signal processing functions for analysis
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks


def compute_fft(signal, fs):

    """
    Compute the FFT of a signal and return the frequencies, magnitude and peaks.
    
    Parameters:
    signal (numpy array): Input signal in time domain.
    fs (float): Sampling frequency.

    
    
    """
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