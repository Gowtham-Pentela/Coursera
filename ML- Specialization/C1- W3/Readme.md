# Logistic Regression with Regularization for Binary Classification

## Overview
This project implements a binary classification model using logistic regression with gradient descent. The model is trained to predict binary outcomes (admission/rejection or acceptance/rejection) based on two features from a dataset. Additionally, L2 regularization is applied to improve model generalization and prevent overfitting. The code includes data preprocessing, cost computation, gradient calculation, and decision boundary visualization.

## Features
- **Binary Classification with Logistic Regression**: Predicts a binary outcome based on two input features.
- **Gradient Descent Optimization**: Uses gradient descent to minimize the cost function.
- **Sigmoid Activation**: The sigmoid function is used to map predictions to probabilities.
- **Regularization**: Adds L2 regularization to the cost function to prevent overfitting.
- **Data Visualization**: Plots decision boundaries and datasets for visualizing model performance.

## Dependencies
- Python 3.7+
- Libraries:
  - `numpy`: for numerical operations
  - `matplotlib`: for data visualization
  - Custom `utils.py`: for helper functions like `plot_data` and `map_feature`

## Files
- **Main Script**: Contains the logistic regression implementation and training.
- **utils.py**: Includes helper functions to load data, map features, and plot decision boundaries.

## Code Details
1. **Data Loading**:
   - Loads datasets (`ex2data1.txt` and `ex2data2.txt`) for binary classification.

2. **Data Preprocessing**:
   - Visualizes data with `plot_data`.
   - Maps features using `map_feature` to create higher-order polynomial terms for improved separability.

3. **Model Implementation**:
   - **Sigmoid Function**: Computes the sigmoid of input `z` to return probabilities.
   - **Cost Function (`compute_cost` and `compute_cost_reg`)**: Calculates logistic regression cost, with and without regularization.
   - **Gradient Calculation (`compute_gradient` and `compute_gradient_reg`)**: Computes gradients of the cost function with respect to model parameters for gradient descent optimization.
   - **Gradient Descent**: Updates weights (`w`) and bias (`b`) using calculated gradients and learning rate.

4. **Training**:
   - Executes gradient descent with specified learning rate (`alpha`), number of iterations (`num_iters`), and regularization parameter (`lambda_`).
   - Evaluates training accuracy and visualizes decision boundaries for both datasets.

5. **Prediction**:
   - **Predict Function**: Predicts the class label by applying a threshold to the model output.

## Usage
1. Clone this repository and navigate to the project directory.
2. Place your datasets (`ex2data1.txt`, `ex2data2.txt`) in the `data` folder.
3. Execute the script:
   ```python
   python logistic_regression.py
   ```
4. The script will output the cost, training accuracy, and plot decision boundaries for each dataset.

## Output
- **Training Accuracy**: Displays the percentage of correct predictions on the training data.
- **Plots**: Shows the data points with the decision boundary, distinguishing between predicted classes.

## Customization
- **Hyperparameters**: Adjust `alpha`, `num_iters`, and `lambda_` to experiment with learning rate, training iterations, and regularization strength.
- **Data**: Replace `ex2data1.txt` and `ex2data2.txt` with other datasets following the same format to use the model on different binary classification problems.

## Notes
- This implementation relies on manually written loops for computing the cost and gradient descent; more efficient implementations may leverage matrix operations.
- Higher `lambda_` values increase regularization strength, which can help with overfitting but may affect model accuracy if set too high.

## Example Output
- **Decision Boundary Plot**: Visualizes the model's decision-making for binary classification.
- **Accuracy**: Indicates the model's performance on training data.
