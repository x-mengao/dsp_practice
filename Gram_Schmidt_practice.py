# This code practices Gram-Schmit Process on IQ data
# This kind of operation often appears in:
#   - IQ imbalance correction
#   - Signal normalization
#   - Baseband signal rotation or scaling

# Steps:
# Once amplitude imbalance Ae and phase imbalance Phie are determined, use Gram-Schmidt method to correct IQ data
#         [     1                 0       ]
# conv =  [                               ]
#         [-tan(Phie)     1/(Ae*cos(Phie))]
# 
# The linear transform takes following matrix multiplication
# [I2]           [I1]
# [  ]  = conv • [  ] <-> I2 = I1, Q2 = -tan(Phie)•I1 + 1/(Ae*cos(Phie)•Q1)
# [Q2]           [Q1]

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


datafile_folderPath = '/Users/xiaom/Documents/Work/Dissertations/Code/SampleData1-motion_measurement_Filename_indicates_motion_displacement_and_gain_of_LNA'
filaname = '40mm_1.txt'
filePath = os.path.join(datafile_folderPath, filaname)
data = pd.read_csv(filePath, delim_whitespace=True, skiprows=2)
data_i = data.iloc[:,1] - np.mean(data.iloc[:,1])
data_q = data.iloc[:,3] - np.mean(data.iloc[:,3])

plt.figure(figsize=(10,6))
plt.scatter(data_i, data_q, label='Original IQ')
plt.title('Raw IQ Plot')
plt.xlabel('Channel I')
plt.ylabel('Channel Q')
plt.xlim(-0.15, 0.15)
plt.ylim(-0.15, 0.15)

# Applying Gram-Schmidt Process
Ae = 1.2
Phie_deg = 20
Phie_rad = Phie_deg / 180 * np.pi

# Build transform matrix that does linear transformation on the IQ data, making the results orthogonal
data_i_imbCorrected = data_i.copy()
data_q_imbCorrected = -np.tan(Phie_rad) * data_i + data_q / (Ae * np.cos(Phie_rad))
plt.scatter(data_i_imbCorrected, data_q_imbCorrected, color='red', label='Imbalance Corrected IQ')
# Takeaway:
# I2 = I1 does not make a new array, changes to I2 affects I1, just makes I2 another name of I1
# I2 = I1.copy() makes a new array copy with same contents, they are independent. 
