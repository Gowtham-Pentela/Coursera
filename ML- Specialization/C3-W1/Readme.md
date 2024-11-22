# K-Means Clustering, Outlier Detection, and Image Compression

This repository provides a comprehensive implementation of K-Means Clustering, Outlier Detection using Gaussian distributions, and an application of K-Means for image compression. Each module is implemented with detailed explanations and visualizations.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Implementation Details](#implementation-details)
4. [Code Overview](#code-overview)
5. [Prerequisites](#prerequisites)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
   - [K-Means Clustering](#k-means-clustering)
   - [Outlier Detection](#outlier-detection)
   - [Image Compression](#image-compression)
8. [Results](#results)
9. [References](#references)
10. [License](#license)

---

## Introduction

This repository is structured into three main sections:
1. **K-Means Clustering**: Groups data points into clusters and visualizes the process.
2. **Outlier Detection**: Uses Gaussian distribution to identify anomalies in data.
3. **Image Compression**: Reduces image color palettes using K-Means, balancing size and quality.

---

## Features

1. **K-Means Clustering**:
   - Visualizes the iterative process of centroid adjustments.
   - Identifies optimal data grouping.

2. **Outlier Detection**:
   - Implements Gaussian anomaly detection.
   - Calculates optimal thresholds using validation sets.
   - Identifies anomalies in both low- and high-dimensional datasets.

3. **Image Compression**:
   - Reduces the number of unique colors in an image while preserving visual quality.

---

## Implementation Details

### K-Means Clustering
- Assigns each data point to its nearest centroid.
- Updates centroid positions iteratively to minimize intra-cluster distance.
- Visualizes the clustering process.

### Outlier Detection
- Estimates feature-wise means and variances.
- Applies multivariate Gaussian probability distribution to detect anomalies.
- Identifies an optimal threshold using F1 scores from a validation set.

### Image Compression
- Uses K-Means to group pixel values into clusters.
- Reconstructs the image using representative colors from cluster centroids.

---

## Code Overview

The implementation includes the following key functions:

### General Setup
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
```

### Data Preparation
```python
X_train, X_val, y_val = load_data()

print("First 5 elements of X_train:\n", X_train[:5])
print("First 5 elements of X_val:\n", X_val[:5])
print("First 5 elements of y_val:\n", y_val[:5])

print("Shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
```

### Data Visualization
```python
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')
plt.title("The First Dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()
```

### Outlier Detection
#### Estimate Gaussian Parameters
```python
def estimate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var

mu, var = estimate_gaussian(X_train)
print("Mean of each feature:", mu)
print("Variance of each feature:", var)
```

#### Identify Anomalies
```python
p = multivariate_gaussian(X_train, mu, var)

def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        F1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1
```

---

## Prerequisites

- **Python 3.x**
- Required libraries:
  - `numpy`
  - `matplotlib`
  - Custom utilities in `utils.py`

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

3. Run the Jupyter Notebook or Python script to explore the features.

---

## Usage

### K-Means Clustering
1. Initialize centroids and call the `run_kMeans` function.
2. Plot the clustering progress.

### Outlier Detection
1. Load datasets with `load_data` and `load_data_multi`.
2. Use `estimate_gaussian` to compute parameters.
3. Run `select_threshold` to determine the best anomaly threshold.

### Image Compression
1. Load an image and preprocess it.
2. Apply K-Means to compress the image.

---

## Results

### Example Outputs:
- **Clustering**: Visual representation of clusters and centroids.
- **Outlier Detection**: Identified anomalies with F1 scores.
- **Image Compression**: Side-by-side comparison of original and compressed images.

---

## References

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- Custom utility functions (`utils.py`)

---

