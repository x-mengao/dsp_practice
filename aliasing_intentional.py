# ##############################################################
# this file explores using alising intentionally to get HR information from a high frequency carrier
# ##############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter

# Signal definition of a HR signal modulated as phase, on carrier radar signal
def create_radar_signal(f_carrier, fs, f_x, duration, do_plot=False):
    t = np.arange(0, duration, 1/fs) # Create time samples
    # Create radar signal with motion modulated phase
    x = np.cos(2 * np.pi * f_x * t)  # Modulated phase due to heartbeat motion
    c = 3e8
    wavelength = c / f_carrier
    radar_signal = np.cos(2 * np.pi * f_carrier * t + 4 * np.pi * x / wavelength)
    # lo_signal = np.cos(2 * np.pi * f_carrier * t)
    lo_signal = np.exp(1j * 2 * np.pi * f_carrier * t)
    freqs_radarSig, X_mag_radarSig, peak_radarSig = run_fft(radar_signal, fs)

    if do_plot:
        fig,ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].plot(t[0:1000], radar_signal[0:1000])
        ax[1].plot(freqs_radarSig, X_mag_radarSig)
        ax[1].plot(freqs_radarSig[peak_radarSig], X_mag_radarSig[peak_radarSig], "x")
        subplot_labels(ax, f'Radar Signal with Modulated Phase at Sample rate {fs/1e9}GHz')
        print(f"Detected peak:{freqs_radarSig[peak_radarSig]}Hz")
        # ax[1].set_xlim(0, fs/2)  # Limit x-axis to positive frequencies

    return radar_signal, lo_signal, t 

def mix_signal(radar_signal, lo_signal, t, fs, do_plot=False):
    # Downconvert the radar signal to baseband
    # mixed_signal = radar_signal * lo_signal
    mixed_signal = lo_signal * np.conjugate(radar_signal)
    freqs_mxed, X_mag_mxed, peak_mxed = run_fft(mixed_signal, fs)

    if do_plot:
        fig,ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].plot(t, np.real(mixed_signal))
        ax[1].plot(freqs_mxed, X_mag_mxed)
        ax[1].plot(freqs_mxed[peak_mxed], X_mag_mxed[peak_mxed], "x")
        print(f"Detected peak:{freqs_mxed[peak_mxed]}Hz")
        subplot_labels(ax, f'Mixed Signal at Sample rate {fs/1e9}GHz')
        # ax[1].set_xlim(0, fs/2)  # Limit x-axis to positive frequencies        
    return mixed_signal

def subplot_labels(ax, title_timeDomain):
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title(title_timeDomain)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Magnitude')
    ax[1].set_title('FFT Magnitude Spectrum')

def run_lpf(order, f_lo, fs, input_signal, do_plot=False):
    f_cutoff = f_lo / (fs/2)
    b, a = butter(order, f_cutoff, 'low')
    filtered_signal = lfilter(b, a, input_signal)

    if do_plot:
        plt.figure(figsize=(10,6))
        plt.plot(filtered_signal)
        plt.title(f'Fitered Signal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
    return filtered_signal

def run_fft(signal, fs):
    N = len(signal)
    X = fft(signal)
    X_magnitude = np.abs(X) / N  # Normalize the magnitude
    frequencies = fftfreq(N, 1/fs)
    peak = np.argmax(X_magnitude)
    return frequencies, X_magnitude, peak

# Signal definition of a HR signal modulated as phase, on carrier radar signal
def create_aliased_signal(f_carrier, f_aliased, f_x, duration, do_plot=False):
    t = np.arange(0, duration, 1/f_aliased) # Create time samples
    # Create radar signal with motion modulated phase
    x = np.cos(2 * np.pi * f_x * t)  # Modulated phase due to heartbeat motion
    c = 3e8
    wavelength = c / f_carrier
    radar_signal_aliased = np.cos(2 * np.pi * f_carrier * t + 4 * np.pi * x / wavelength)
    # if_lo_signal = np.cos(2 * np.pi * np.abs(f_carrier - f_aliased) * t)
    if_lo_signal = np.exp(1j * 2 * np.pi * np.abs(f_carrier - f_aliased) * t)
    freqs_radarSig_aliased, X_mag_radarSig_aliased, peak_radarSig_aliased = run_fft(radar_signal_aliased, f_aliased)

    if do_plot:
        fig,ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].plot(t, radar_signal_aliased)
        ax[1].plot(freqs_radarSig_aliased, X_mag_radarSig_aliased)
        ax[1].plot(freqs_radarSig_aliased[peak_radarSig_aliased], X_mag_radarSig_aliased[peak_radarSig_aliased], "x")
        subplot_labels(ax, f'Radar Signal with Modulated Phase at Sample rate {f_aliased/1e9}GHz')
        print(f"Detected peak:{freqs_radarSig_aliased[peak_radarSig_aliased]}Hz")
        # ax[1].set_xlim(0, fs/2)  # Limit x-axis to positive frequencies

    return radar_signal_aliased, if_lo_signal, t

# User defined parameters
f_hr = 3
# If defined for radar perspective
f_carrier = 2.4e9
sample_rate = f_carrier * 10
aliased_sample_rate = 2.400000100e9
t_duration = 2e-3

# IF defined for ppg perspective
# f_carrier = 2.4e3
# sample_rate = f_carrier * 10
# aliased_sample_rate = 2.5e3
# t_duration = 20

# Main functions calling
radar_signal, lo_signal, t_radarSig = create_radar_signal(f_carrier, sample_rate, f_hr, t_duration, do_plot=True)
# mixed_signal = mix_signal(radar_signal, lo_signal, t_radarSig, sample_rate, do_plot=True)
# filtered_signal = run_lpf(4, 100, sample_rate, mixed_signal, do_plot=True)
radar_signal_aliased, if_lo_signal, t_radarSigAliased = create_aliased_signal(f_carrier, aliased_sample_rate, f_hr, t_duration, do_plot=True)
# mixed_aliased_signal = mix_signal(radar_signal_aliased, if_lo_signal, t_radarSigAliased, aliased_sample_rate, do_plot=True)

