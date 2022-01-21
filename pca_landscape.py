# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:57:50 2022

@author: Emmanuel Calvet
"""
# !pip install opencv-python
# !pip install glob

import numpy as np
import matplotlib.pyplot as plt
import cv2 

import glob

import sys
from pathlib import Path


# Get path for files
path_root = str(Path(__file__).parents[0])

# %% import database
filelists = glob.glob(path_root+'/dataset_landscape/*.jpg')
original_nb_image = len(filelists)
m = int(original_nb_image/8) # Reduce the size to compute faster
l = 800 # Length of images
n = l*l # Number of pixels


# Create X matrix
X = []
for i, filelist in enumerate(filelists[:m]):
    # Open image in an array
    image = cv2.imread(filelist)
    image_shape = image.shape
    # convert to grayscale (removes 2*800*800 dimensions)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image (needed to create the matrix)
    resized_image_gray = cv2.resize(image_gray, (l, l))
    # Reshape the image into a vector
    X.append(resized_image_gray.reshape(resized_image_gray.size))

# Reshape to get a [n * m] matrix    
X = np.array(X).T
    
# %% SVD

from scipy.linalg import svd

# Compute the SVD 
U,S,V_T = svd(X, full_matrices=False)

# %% Plot

# Eigenlandscape
nb_plots = 20 # Number of eigenlandscape in plot
nb_col = 5 # Number of plot per columns
rest = nb_plots % nb_col  # if ever
nb_rows = int(nb_plots/nb_col) + rest # Number of plot per rows

fig = plt.figure()
for i, idx in enumerate(range(0, nb_plots)):
    
    # Get the eigenlandscape
    eigen_landscape = U[:, idx]
    eigen_landscape = eigen_landscape.reshape(l, l)
    
    # Plot
    ax = fig.add_subplot(nb_rows, nb_col, i+1)    
    ax.set_title(f'Eigen-landscape #{idx+1}')
    ax.imshow(eigen_landscape)
plt.tight_layout()

# %% PCA

# Get the third first principal components and plot them
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
for j in range(X.shape[1]):
    x = U[:, 0] @ X[:, j].T # PC1
    y = U[:, 1] @ X[:, j].T # PC2
    z = U[:, 2] @ X[:, j].T # PC3
    
    ax1.scatter(x, y, z, s=40)
    
ax1.set_xlabel('PC1', fontsize=18)
ax1.set_ylabel('PC2', fontsize=18)
ax1.set_zlabel('PC3', fontsize=18)
ax1.set_title('The first 3 principal components', fontsize=20)
plt.tight_layout()
