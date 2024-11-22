# Neural Networks for Multiclass classification

## Overview

This repository contains a deep learning model designed for digit classification using TensorFlow and Keras. The model is trained to recognize handwritten digits from a dataset, such as the MNIST dataset, using a simple feedforward neural network architecture. The code demonstrates the use of softmax activation for the output layer, visualization of input data, and evaluation of model performance.

## Requirements

The following Python libraries are required to run this project:

- `numpy` (>= 1.21.0)
- `tensorflow` (>= 2.8.0)
- `matplotlib` (>= 3.4.0)
- `autils` (a custom utility module)
- `lab_utils_softmax` (a custom utility module)
- `public_tests` (a custom testing module)

To install the necessary dependencies, use:

```bash
pip install numpy tensorflow matplotlib
```

You will also need to ensure the availability of `autils`, `lab_utils_softmax`, and `public_tests` modules, which are custom components used in the project.

## File Structure

The file contains the following sections:

1. **Imports and Setup**: Libraries like TensorFlow, numpy, and matplotlib are imported, followed by the setup of logging and plotting configurations.
2. **Softmax Function**: A custom implementation of the softmax function is defined to demonstrate how softmax works for converting logits into probabilities.
3. **Data Loading and Visualization**: The dataset (`X`, `y`) is loaded, and random examples are displayed using `matplotlib`.
4. **Model Architecture**: A simple feedforward neural network is created using TensorFlow/Keras, with three layers and ReLU activation for hidden layers and linear activation for the output layer.
5. **Model Training**: The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss function. It is then trained for 40 epochs.
6. **Prediction and Evaluation**: The model is evaluated on random samples, and predictions are made on test images, displaying the predicted label alongside the true label.
7. **Error Analysis**: The errors in the model predictions are calculated and displayed.

## Usage

### Loading Data

The dataset (`X` and `y`) is loaded via the `load_data()` function, and the shapes of `X` and `y` are printed. These contain the pixel data for the images and their corresponding labels, respectively.

```python
X, y = load_data()
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
```

### Visualizing Data

The code generates a plot of 64 random images from the dataset, reshaped to 20x20 pixels. The true label for each image is displayed above it.

```python
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    ax.imshow(X_random_reshaped, cmap='gray')
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
fig.suptitle("Label, image", fontsize=14)
```

### Defining the Model

The model is defined as a sequential neural network using the Keras API. It contains three layers:
- Input layer with 400 nodes (for flattened 20x20 images)
- Two hidden layers with 25 and 15 nodes, respectively, using ReLU activation
- Output layer with 10 nodes (corresponding to the digits 0-9) using linear activation

```python
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='linear')
    ], name="my_model"
)
```

### Training the Model

The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss function. It is trained for 40 epochs on the dataset.

```python
history = model.fit(X, y, epochs=40)
```

### Prediction

After training, the model makes predictions on individual test samples. The results are printed, showing both the raw output vector and the softmax probabilities.

```python
prediction = model.predict(image_of_two.reshape(1, 400))
prediction_p = tf.nn.softmax(prediction)
```

### Error Analysis

The model’s accuracy is evaluated by comparing its predictions to the true labels. A summary of errors is printed for the dataset.

```python
print(f"{display_errors(model, X, y)} errors out of {len(X)} images")
```

### Softmax Implementation

The `my_softmax` function implements the softmax operation manually to demonstrate the mathematical concept. It compares the result with TensorFlow’s built-in softmax function.

```python
def my_softmax(z):
    N = len(z)
    a = np.zeros(N)
    ez_sum = 0
    for k in range(N):
        ez_sum += np.exp(z[k])
    for j in range(N):
        a[j] = np.exp(z[j]) / ez_sum
    return a
```

### Custom Utility Functions

The repository includes several custom utility functions such as `test_my_softmax`, `plt_softmax`, and `display_digit`, which help in testing, visualizing, and displaying the model's predictions.

## Results

After running the code, you will see:
1. The model summary with layer details.
2. A plot of random images from the dataset.
3. Model performance after training, including error statistics.
4. A display of predictions and probabilities for individual test images.

## Notes

- The code uses the `matplotlib` library for visualizations, so ensure your environment supports interactive plots (e.g., Jupyter Notebook).
- The dataset (`X`, `y`) is assumed to be pre-loaded, and you may need to modify the `load_data()` function if using a custom dataset.
- Warnings related to future versions of libraries are suppressed to avoid cluttering the output.

## Conclusion

This project demonstrates a simple deep learning workflow using TensorFlow/Keras for digit classification. The model architecture is flexible, and you can modify it to improve accuracy or explore different activation functions and training strategies.