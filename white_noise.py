# This script models white noise 

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# Generate 1000 samples of Gaussian white noise
# mean = 0, std = 1 (variance = 1)

white_noise = np.random.normal(0, 1, 1000)
fig, ax = plt.subplots(1,2, figsize=(10, 6))
# Option 1 - plot noise
ax[0].plot(white_noise)
ax[0].set_title("Simulate white noise")
ax[0].set_xlabel("Time sample")
ax[0].set_ylabel("Amplitude")
# Option 2 - plot histogram
ax[1].hist(white_noise, bins=30)
ax[1].set_title("Histogram of white noise")
ax[1].set_xlabel("value")
ax[1].set_ylabel("Occurance")

plt.show()

plot_acf(white_noise, lags=20)
plt.title("Autocorrelation Function of White Noise")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()
# The autocorrelation plot for white noise should show a significant spike at lag 0 
# (correlation with itself is 1) and then values close to zero for all other lags, 
# ideally within the confidence bounds.