import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
fs = 100.0  # Sampling frequency (Hz)
duration = 1.0  # Total duration of the signal (s)
f = 5.0      # Frequency of the cosine wave (Hz)
A = 1.0      # Amplitude
tau = 0.25   # Time delay (s)

# --- Generate Time Samples ---
num_samples = int(fs * duration)
t = np.linspace(0, duration, num_samples, endpoint=False)

# --- Calculate the Delayed Time ---
t_delayed = t - tau

# --- Evaluate the Signal at the Original Time Samples using the Delayed Time ---
y0 = A * np.cos(2 * np.pi * f * t)
y = A * np.cos(2 * np.pi * f * t_delayed)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(t, y0, label='Original Signal', linestyle='--')
plt.plot(t, y, label='Delayed Signal', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Cosine Wave with Delay (tau = {tau} s)")
plt.grid(True)
plt.legend()
plt.show()