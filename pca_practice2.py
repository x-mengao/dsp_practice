# This is a practice code for PCA, using scikit-learn library
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
# Option 1 - Generate data
x = np.array([2.5, 0.5, 2.2, 1.9, 3.1])
y = np.array([2.4, 0.7, 2.9, 2.2, 3.0])
data = np.column_stack((x, y)) # stack the vectors into 2D array (row: observations, columns: features)
# Option 2 - Load data in

# # Center the data
# data_centered = data - np.mean(data, axis=0)

# Initialize PCA with number of components, i.e. 2 principal components
pca = PCA(n_components=2)   
projected_data = pca.fit_transform(data)    # PCA automatically centers data, no need additional centering

# Access the principal components and explained variance
components = pca.components_


# Now, projected_data contains the representation of the original data in the space defined by the principal components, 
# with the first column corresponding to the direction of maximum variance.

plt.figure(figsize=(10, 6))
plt.scatter(projected_data[:,0], projected_data[:,1], color='skyblue', edgecolor='k', s=100)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

