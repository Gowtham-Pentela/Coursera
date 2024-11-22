# Advice for Applying Machine Learning

This project explores multiple machine learning techniques to model data, including polynomial regression, regularized linear regression, and neural networks using TensorFlow. The purpose is to evaluate model performance on a synthetic dataset, with a focus on optimization, regularization, and comparison of various model types.

## Project Overview
This notebook demonstrates:

- **Data generation** and splitting
- **Linear regression** with and without regularization
- **Polynomial regression** to model complex relationships
- **Neural networks** for classification with different architectures
- **Model evaluation** using Mean Squared Error (MSE) and Categorization Error (CE)

The goal is to showcase the application of machine learning techniques to fit data, tune model parameters, and evaluate performance.

## Requirements
Before running the notebook, make sure you have the following libraries installed:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

## Key Modules and Functions

### 1. Data Generation
```python
X, y, x_ideal, y_ideal = gen_data(18, 2, 0.7)
```
Generates synthetic data for regression tasks, including training and testing sets.

### 2. Model Evaluation
- **Mean Squared Error (MSE):**
  The function `eval_mse` calculates the mean squared error between actual and predicted values.
  
  ```python
  def eval_mse(y, yhat):
      m = len(y)
      err = 0.0
      for i in range(m):
          err += (y[i] - yhat[i])**2
      err = (1/(2*m)) * err
      return err
  ```

- **Categorization Error (CE):**
  The function `eval_cat_err` calculates the categorization error for classification models.
  
  ```python
  def eval_cat_err(y, yhat):
      m = len(y)
      incorrect = 0
      for i in range(m):
          if y[i] != yhat[i]:
              incorrect += 1
      cerr = (1/m) * incorrect    
      return cerr
  ```

### 3. Model Architectures
- **Polynomial Linear Regression:**
  Uses `PolynomialFeatures` to model nonlinear relationships between features and target variables.
  
- **Neural Networks:**
  Uses `Sequential` models from TensorFlow Keras, with various architectures like simple, complex, and regularized models.

### 4. Training and Testing Models
- Models are trained using TensorFlow, with various configurations such as complex, simple, and regularized models. The models are trained over 1000 epochs with a learning rate of `0.01`.
  
  ```python
  model = Sequential([
      Dense(120, activation='relu'),
      Dense(40, activation='relu'),
      Dense(6, activation='linear')
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(0.01)
  )
  model.fit(X_train, y_train, epochs=1000)
  ```

### 5. Regularization
Regularization is applied using L2 regularization (Ridge) to prevent overfitting.

```python
model_r = Sequential([
    Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(6, activation='linear')
])
```

### 6. Hyperparameter Tuning
The models are evaluated for different values of `lambda` (regularization strength), and the optimal regularization parameter is chosen by evaluating the test error.

```python
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
```

### 7. Visualizations
The project includes several visualizations for data exploration and model evaluation, such as:
- Plotting training, test, and ideal values.
- Plotting the fit of polynomial regression and neural network predictions.
- Comparing different models' performance (simple vs. complex, regularized vs. non-regularized).

```python
plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
```

### 8. Results
- **Training and testing errors** are calculated and printed after training the models.
- **Categorization errors** are calculated for classification tasks to evaluate model performance.

### 9. Model Comparison
The models are compared by plotting the categorization error for training, validation, and test sets.

```python
plt_compare(X_test, y_test, classes, model_predict_s, model_predict_r, centers)
```

## Usage Instructions

1. **Run the notebook** after installing the required libraries. Make sure you have a compatible Python environment.
2. **Modify the data generation** settings or replace with your dataset if necessary.
3. **Tweak model architectures**, regularization parameters, and training epochs to suit your use case.
4. **Evaluate model performance** using the provided functions and visualizations to assess training and test errors.

## Conclusion
This project demonstrates a broad range of machine learning techniques applied to a synthetic dataset, including regression, classification, and neural network models. The use of TensorFlow and scikit-learn provides a solid foundation for building and evaluating machine learning models. Regularization and hyperparameter tuning techniques ensure the models are optimized for generalization.

