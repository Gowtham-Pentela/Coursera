# Neural Networks for Binary Classification

## Overview

This repository contains an implementation of a simple neural network to classify handwritten digits using a dataset. The neural network is built with TensorFlow and Keras, and it includes the implementation of both TensorFlow-based and custom Numpy-based layers to understand and predict the images. The goal of this project is to provide an educational walkthrough of the steps involved in building and training a neural network for image classification, as well as to highlight the differences between TensorFlow's built-in operations and custom implementation.

## Key Components

### 1. Data Loading and Exploration
- The dataset is loaded using the `load_data()` function, which returns input features `X` and corresponding labels `y`. The dataset consists of 400 features per image, each representing a 20x20 grayscale image of a digit.
- The dataset is visualized using `matplotlib` to show some random images from the dataset along with their corresponding labels.

### 2. Model Building
- A simple feed-forward neural network is constructed using TensorFlow/Keras:
    - **Input Layer**: 400 features (reshaped 20x20 images).
    - **Hidden Layer 1**: 25 units with sigmoid activation.
    - **Hidden Layer 2**: 15 units with sigmoid activation.
    - **Output Layer**: 1 unit with sigmoid activation (binary classification).
- The model is compiled with binary cross-entropy loss and Adam optimizer.

### 3. Training and Prediction
- The model is trained on the dataset for 20 epochs.
- Predictions are made on specific inputs (e.g., the first and 500th examples), and results are processed using a threshold to classify the predictions as either 0 or 1.

### 4. Custom Numpy-based Neural Network
- A custom `my_dense()` function is implemented using Numpy to simulate a dense layer with a specified activation function (sigmoid in this case).
- The custom network is used in the `my_sequential()` function, which mimics the TensorFlow model's architecture but operates entirely using Numpy functions.

### 5. Visualization of Results
- The model's predictions are visualized using `matplotlib`:
    - Random images from the dataset are displayed, and both TensorFlow-based and custom Numpy-based predictions are compared.
    - Misclassified images are highlighted, showing the predicted label and true label.
    
### 6. Matrix Operations
- Various matrix operations such as element-wise addition and multiplication are demonstrated with Numpy to show how broadcasting works with matrices and vectors.

## Setup

1. **Dependencies**
   - Python 3.x
   - TensorFlow (for building and training the neural network)
   - Numpy (for matrix operations and custom layer implementation)
   - Matplotlib (for visualizing the dataset and predictions)
   - Custom utility functions (`autils`)

2. **Installation**
   Install the required libraries using pip:
   ```bash
   pip install numpy tensorflow matplotlib
   ```

3. **Data**
   The `load_data()` function is used to load the dataset, which is assumed to be preprocessed and available. If you need to use a different dataset, modify the `load_data()` function accordingly.

## Code Usage

1. **Model Training**:
   After loading the data, the neural network is built and trained using the following code:
   ```python
   model.fit(X, y, epochs=20)
   ```

2. **Prediction**:
   Predictions are made using the trained model, as shown below:
   ```python
   prediction = model.predict(X[0].reshape(1, 400))
   ```

3. **Custom Numpy Model**:
   For understanding the internal workings of a neural network, the `my_dense()` and `my_sequential()` functions allow you to implement a neural network using just Numpy operations.

## Example Outputs

- **Predictions**:
    - Example 1: Prediction of the first image (digit 0):
      ```python
      prediction = model.predict(X[0].reshape(1, 400))
      print(f"Prediction for 0: {prediction}")
      ```

    - Example 2: Prediction of a random image (digit 1):
      ```python
      prediction = model.predict(X[500].reshape(1, 400))
      print(f"Prediction for 1: {prediction}")
      ```

- **Visualizing Misclassifications**:
    - Display images that were misclassified along with the correct label and predicted label.
      ```python
      fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
      ```

## Conclusion

This project demonstrates how a simple neural network can be implemented using TensorFlow and Keras for digit classification. Additionally, the implementation of custom layers in Numpy helps understand the inner workings of neural networks at a lower level. By comparing predictions from TensorFlow and Numpy-based implementations, the project provides insight into the processes that occur during neural network training and inference.
