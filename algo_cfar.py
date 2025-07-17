# ##############################        
# This snippet is simulating a CFAR detection algorithm
# Copyright (C) 2025  Xiaomeng Gao
# 
# Key Takeaway: 
# 1. CFAR is used to detect real signal in noisy data, while controlling the rate of false alarm. 
# CFAR sets a dynamic threshold that adapts to the local noise level. 
# 2. In CA-CFAR implementation, the constant false alarm rate is implicitly by the threshold_scaling 
# factor, in combination with cell averaging machenism. 
# 3. The threshold_scaling factor, derives from alpha = N * (P_FA^(-1/N) - 1), 
#   where N is number of ref_cells, P_FA is desired false alarm probability (e.g. 1e-3)
#   e.g. when N=8, P_FA=1e-3, alpha = 8 * (1e-3^(-1/8) - 1) = 10.97 ~=11


# 
# Steps
# 1. For each bin (cell under test), CFAR looks at a guard band then collect data from ref cells around 
# bin to estimate local noise
# 2. Computes a threshold based on statistics
# 3. Compare test cell's amplitude to this threshold
# 4. If test cell > threshold → Declare a detection.
# ##############################   

import numpy as np
import matplotlib.pyplot as plt

# Simulate 1D range data with noise + 3 targets
np.random.seed(42)
signal = np.random.randn(100) * 2 # Noise simulation
signal[20] = 20 # Target 1 simulation
signal[50] = 15 # Target 2 simulation
signal[75] = 18 # Target 3 simulation

# CFAR parameters
num_guard = 2           # number of guard cells on each side of CUT
num_ref = 8             # number of reference cells to estimate noise
cfar_probability = 1e-3
threshold_scale = 2 * num_ref * (cfar_probability**(-1 / (2 * num_ref)))
# threshold_scale = 3.5   # sensitivity tuning, also known as proxy of P_FA. Higher = stricter, few detections

cfar_output = np.zeros_like(signal)
threshold_line = np.zeros_like(signal)

for i in range(num_ref + num_guard, len(signal) - num_ref - num_guard): 
    # sweeping bins from first bin that has enough ref + guard bins on the left
    start = i - num_guard - num_ref
    end = i + num_guard + num_ref + 1
    ref_cells = np.concatenate([signal[start:i - num_guard], signal[i + num_guard + 1 :end]])

    noise_level = np.mean(ref_cells)
    threshold = noise_level * threshold_scale
    threshold_line[i] = threshold

    if signal[i] > threshold:
        cfar_output[i] = signal[i]  # when no detection made, output 0; if exceed threshold, original signal value pass to output.

is_detected = np.abs(cfar_output) > 0
not_detected = cfar_output == 0
x = np.arange(len(cfar_output))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(signal, label='Input Signal')
plt.plot(x[is_detected], cfar_output[is_detected], 'g*', label='Detected')
plt.plot(x[not_detected], cfar_output[not_detected], 'r*', label='Not detected')
plt.plot(threshold_line, color='gray', linestyle='--', label='Adaptive Threshold')
plt.title("CA-CFAR Detection (with Leakage and Adjacent Bins)")
plt.xlabel("Range Bin")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()