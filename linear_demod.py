# This is a practice script for linear demodulation
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


datafile_folderPath = '/Users/xiaom/Documents/Work/Dissertations/Code/SampleData1-motion_measurement_Filename_indicates_motion_displacement_and_gain_of_LNA'
filaname = '40mm_1.txt'
filePath = os.path.join(datafile_folderPath, filaname)
data = pd.read_csv(filePath, delim_whitespace=True, skiprows=2)
data_i = data.iloc[:,1]
data_q = data.iloc[:,3]

plt.figure(figsize=(10,6))
plt.scatter(data_i, data_q)
plt.title('Raw IQ Plot')
plt.xlabel('Channel I')
plt.ylabel('Channel Q')

data_vec = np.column_stack((data_i, data_q))

# Method 1 - Initialize PCA with number of components, i.e. 2 principal components
# pca = PCA(n_components=2)
# projected_data = pca.fit_transform(data_vec)

# Method 2 - Covariance Matrix and Eigenvectors
data_centered = data_vec - np.mean(data_vec)
cov_matrix = np.cov(data_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigen values so that the principal components are ordered by the amount of variance they capture.
sorted_indicies = np.argsort(eigenvalues)[::-1] # the reverse order makes the max variance fisrt element in eigenvalues
eigenvalues = eigenvalues[sorted_indicies]
eigenvectors = eigenvectors[:, sorted_indicies]

projected_data = np.dot(data_centered, eigenvectors)

plt.figure(figsize=(10, 6))
plt.scatter(projected_data[:,0], projected_data[:,1], color='skyblue')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

