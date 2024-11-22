# Decision Tree Classifier for Mushroom Classification

This project demonstrates the process of building a decision tree classifier to predict whether a mushroom is edible or poisonous based on various features. The dataset contains several binary features, each describing different characteristics of a mushroom, such as its cap color, shape, and the presence of a ring.

The goal is to construct a decision tree that splits the dataset at each node based on the feature that provides the highest information gain, recursively building the tree until a specified maximum depth is reached.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Explanation](#code-explanation)
4. [Functions](#functions)
5. [Testing](#testing)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

Ensure you have the required dependencies installed to run the code:

```bash
pip install numpy matplotlib
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/mushroom-decision-tree.git
cd mushroom-decision-tree
```

2. Open the `mushroom_classifier.py` file and run the script in your Python environment:

```bash
python mushroom_classifier.py
```

---

## Code Explanation

### 1. Data Preparation
The dataset `X_train` contains features representing various properties of the mushrooms, and `y_train` contains labels indicating whether each mushroom is edible (`1`) or poisonous (`0`). The shape and size of the dataset are printed to give insight into its structure.

```python
X_train = np.array([[1,1,1], [1,0,1], [1,0,0], [1,0,0], [1,1,1], [0,1,1], [0,0,0], [1,0,1], [0,1,0], [1,0,0]])
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
```

### 2. Entropy Calculation
The entropy function computes the uncertainty or impurity of the dataset at a given node, based on the proportion of edible and poisonous mushrooms.

```python
def compute_entropy(y):
    ...
    return entropy
```

### 3. Data Splitting
The function `split_dataset` splits the dataset into two subsets (left and right) based on the value of a specified feature. For example, the `0th` feature indicates whether the mushroom has a brown cap or not.

```python
def split_dataset(X, node_indices, feature):
    ...
    return left_indices, right_indices
```

### 4. Information Gain
The function `compute_information_gain` computes the information gain for a particular feature at a node. This is used to determine how well a feature splits the dataset into two homogeneous groups (edible vs. poisonous).

```python
def compute_information_gain(X, y, node_indices, feature):
    ...
    return information_gain
```

### 5. Best Split Selection
`get_best_split` finds the feature that provides the highest information gain and selects it as the best feature to split the node on.

```python
def get_best_split(X, y, node_indices):
    ...
    return best_feature
```

### 6. Tree Construction
`build_tree_recursive` is the main function that recursively constructs the decision tree by splitting the dataset based on the best feature at each node, until the maximum depth is reached.

```python
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    ...
```

---

## Functions

### `compute_entropy(y)`
Computes the entropy of a dataset based on the class labels `y`. The entropy quantifies the impurity or uncertainty in the dataset at a given node.

### `split_dataset(X, node_indices, feature)`
Splits the dataset into two subsets based on the value of the specified feature. The left subset contains samples where the feature value is `1`, and the right subset contains samples where the feature value is `0`.

### `compute_information_gain(X, y, node_indices, feature)`
Calculates the information gain from splitting the node on the given feature. Information gain is used to decide which feature to use for splitting the dataset at each node.

### `get_best_split(X, y, node_indices)`
Finds the feature that maximizes the information gain and returns the index of that feature.

### `build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth)`
Builds the decision tree recursively by splitting the dataset at each node. The depth of the tree is controlled by the `max_depth` parameter.

---

## Testing

The provided tests ensure the correctness of key functions, including:
- **Entropy calculation** (`compute_entropy_test`)
- **Data splitting** (`split_dataset_test`)
- **Information gain computation** (`compute_information_gain_test`)
- **Best split selection** (`get_best_split_test`)

The tests are designed to check if the functions correctly compute entropy, split data, and calculate information gain.

---

## Results

The decision tree is built and visualized at each step, showing how the dataset is split based on the best feature. The tree structure is printed in the console, displaying the depth, feature used for splitting, and the resulting child nodes.

Additionally, a visualization of the tree split is generated using `generate_split_viz` and `generate_tree_viz`, providing a graphical representation of the decision tree.

---

## Contributing

Contributions are welcome! If you find a bug or want to improve the project, please feel free to fork the repository and create a pull request. 

To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new pull request.

---