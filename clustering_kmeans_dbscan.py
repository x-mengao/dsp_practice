# #############################################
# # This is the pracice code for clustering
# Tasks:
# 1. Generate 2D mock radar point cloud data representing an occupancy scene (e.g. two groups of points = 2 zones).
# 2. Apply K-Means (with k=2) and DBSCAN to cluster.
# 3. Visualize clusters using different colors.
# #############################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# Generate mock 2D point cloud data
np.random.seed(0)   # for reproducibility, i.e. each time generate same random number so others can repeat same result
cluster_1 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2)) # centered at (5,5)
cluster_2 = np.random.normal(loc=[7, 7], scale=0.5, size=(50, 2))
# Add random noise points uniformly spread
noise = np.random.uniform(low=3, high=8, size=(20, 2))  # 20 noisy points in the region

point_cloud = np.vstack([cluster_1, cluster_2, noise])

# Apply K-means
# label indicates the cluster each data point belongs to
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(point_cloud) # if n_clusters = 2, returned labels are cluster # 0 or 1 for each point

# Apply DBSCAN
# eps or epsilon is max distance beween 2 points, think as radius
# min_samples is min number of neighboring points req to form a dense region, i.e. a core point
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(point_cloud) # returned labels indicate cluster each data point belongs to, means noise if label=-1

# Plotting in subplots
fig, ax = plt.subplots(1, 3, figsize=(10, 8))
# Oringinal point cloud plot
ax[0].scatter(cluster_1[:,0], cluster_1[:,1], color='red')
ax[0].scatter(cluster_2[:,0], cluster_2[:,1], color='blue')
ax[0].scatter(noise[:,0], noise[:,1], color='black')
ax[0].set_title('Original point cloud cluster')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')

# K-Means plot
ax[1].scatter(point_cloud[:,0], point_cloud[:,1], c=kmeans_labels, cmap='tab10')
ax[1].set_title('K-Means Clustering (k=2)')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')

# DBSCAN plot
ax[2].scatter(point_cloud[:,0], point_cloud[:,1], c=dbscan_labels, cmap='tab10')
ax[2].set_title('DBSCAN Clustering')
ax[2].set_xlabel('X')
ax[2].set_ylabel('Y')

plt.tight_layout()
plt.show()
