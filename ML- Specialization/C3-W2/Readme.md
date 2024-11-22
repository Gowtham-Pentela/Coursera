
# Movie Recommendation System

This project builds a movie recommendation system using collaborative filtering and content-based techniques. It demonstrates how to predict a user's ratings for movies that they haven't rated yet, based on previous ratings and user preferences.

## Overview

The core idea of this recommendation system is to predict movie ratings for an individual user, based on the ratings of other users and the features of the movies. It uses a combination of matrix factorization (for collaborative filtering) and regularization techniques to minimize overfitting.

### Key Concepts:
- **Collaborative Filtering**: Recommends items based on the preferences of other users who have similar tastes.
- **Matrix Factorization**: Decomposes the user-item matrix into smaller matrices to find latent factors (e.g., user preferences and item features).
- **Regularization**: Prevents overfitting by adding a penalty for large values in the feature matrices.

## Setup

1. **Dependencies**:  
   The following libraries are required to run the project:
   - TensorFlow
   - NumPy
   - Pandas

2. **Preprocessed Data**:  
   The project assumes the existence of preprocessed data, which includes:
   - `Y`: A matrix of user ratings for movies.
   - `R`: A binary matrix indicating whether a user has rated a movie.

3. **Files Required**:  
   - `recsys_utils.py`: Contains utility functions for loading and preprocessing data, as well as for the cost function.
   - `public_tests.py`: Contains tests for validating the implementation.
   - `movieList.csv`: A list of movies with information such as title, average rating, and the number of ratings.

## Code Walkthrough

### Data Loading

The initial part of the code loads pre-calculated parameters and the rating data:

```python
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()
```

- `X`: Movie features.
- `W`: User preferences.
- `b`: Bias terms for users.
- `Y`: Ratings matrix.
- `R`: Binary matrix indicating whether a user rated a movie.

### Cost Function

The `cofi_cost_func` function calculates the cost of the recommendation system by comparing the predicted ratings with the actual ratings and applying regularization:

```python
def cofi_cost_func(X, W, b, Y, R, lambda_):
    # Implement the cost calculation logic
    # Sum squared errors for rated movies and apply regularization
```

This function is later vectorized for speed using TensorFlow's operations in `cofi_cost_func_v`.

### Training the Model

The model is trained using gradient descent to minimize the cost function. TensorFlow's `GradientTape` is used for automatic differentiation:

```python
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

### Predictions

Once the model is trained, predictions are made by calculating the dot product of the user preferences (`W`) and movie features (`X`), and adding the user bias (`b`):

```python
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = p + Ymean
```

Predictions are then sorted to recommend the highest-rated movies that the user has not rated yet.

### Output

The program will print predictions for the user, as well as the original vs. predicted ratings for the movies that the user has already rated:

```python
for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')
```

Additionally, the recommended movies are displayed based on predicted ratings.

## Evaluation

The model's performance can be evaluated by comparing the predicted ratings to the actual ratings provided by the user.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas

### Running the Code

To run the recommendation system, simply execute the script:

```bash
python recommendation_system.py
```

This will load the data, train the model, and output movie recommendations based on the user's ratings.

## Conclusion

This recommendation system demonstrates how to implement matrix factorization using TensorFlow for collaborative filtering. It provides a personalized list of movie recommendations by predicting ratings for unseen movies based on the user's preferences.

---
