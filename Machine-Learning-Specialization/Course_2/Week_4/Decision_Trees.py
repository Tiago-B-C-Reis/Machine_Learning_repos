import numpy as np
import matplotlib.pyplot as plt
from Package_1 import utils

X_train = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1],
                    [0, 1, 0], [1, 0, 0]])
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:", type(X_train))
print("First few elements of y_train:", y_train[:5])
print("Type of y_train:", type(y_train))
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
print("\n")


# UNQ_C1
# GRADED FUNCTION: compute_entropy
def compute_entropy(y):
    """
    Computes the entropy for
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
    Returns:
        entropy (float): Entropy at that node
    """
    # You need to return the following variables correctly
    entropy = 0.

    if len(y) != 0:
        p_1 = 0
        for i in y[y == 1]:
            p_1 += i
        p1 = p_1 / len(y)
        p0 = 1 - p1
        # For p1 = 0 and 1, set the entropy to 0 (to handle 0log0)
        if p1 != 0 and p1 != 1:
            # Your code here to calculate the entropy using the formula provided above
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        else:
            entropy = 0.

    return entropy
# ----------------------------------------------------------------------------------------


# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"
print("Entropy at root node: ", compute_entropy(y_train))


# GRADED FUNCTION: split_dataset
def split_dataset(X, node_indices, feature):
    """Splits the data at the given node into
    left and right branches
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0"""

    left_indices = []
    right_indices = []

    if 0 <= feature <= 3:
        for i in node_indices:
            if X[i][feature] == 1:
                left_indices.append(i)
            elif X[i][feature] == 0:
                right_indices.append(i)
    else:
        print("\nPlease enter an integer between 0 and 2 for 'feature'.")

    return left_indices, right_indices


""" How I have done it before take a look on the solutions to see if it was more efficient:
    (...)
    z = 0
    if 0 <= feature <= 3:
        for i in X[0:, feature]:
            if i == 1:
                left_indices.append(node_indices[z])
            elif i == 0:
                right_indices.append(node_indices[z])
            z += 1
    (...)
"""
# ----------------------------------------------------------------------------------------

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0
left_indices, right_indices = split_dataset(X_train, root_indices, feature)
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)
# Visualize the split
utils.generate_split_viz(root_indices, left_indices, right_indices, feature)


# UNQ_C3
# GRADED FUNCTION: compute_information_gain
def compute_information_gain(X, y, node_indices, feature):
    """Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed"""
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)

    weighted_entropy = (w_left * left_entropy) + (w_right * right_entropy)
    information_gain = node_entropy - weighted_entropy

    return information_gain


""" How I have done it before take a look on the solutions to see if it was more efficient:
    (...)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    information_gain = compute_entropy(y_node) - ((w_left * compute_entropy(y_left) + 
                                                  (w_right * compute_entropy(y_right)))
    (...)
"""
# ----------------------------------------------------------------------------------------


info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)
info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)


# UNQ_C4
# GRADED FUNCTION: get_best_split
def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    Returns:
        best_feature (int):     The index of the best feature to split"""
    # Some useful variables
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0

    # Iterate through all features
    for feature in range(num_features):
        # Your code here to compute the information gain from splitting on this feature
        info_gain = compute_information_gain(X, y, node_indices, feature)
        # If the information gain is larger than the max seen so far
        if info_gain > max_info_gain:
            # Your code here to set the max_info_gain and best_feature
            max_info_gain = info_gain
            best_feature = feature
    return best_feature


""" How I have done it before take a look on the solutions to see if it was more efficient:
(...)
    feature_information_gain = []

    feature = 0
    while feature < num_features:
        information_gain_i = compute_information_gain(X, y, node_indices, feature)
        feature_information_gain.append(information_gain_i)
        feature += 1

    max_info_gain = max(feature_information_gain)
    best_feature_index = feature_information_gain.index(max_info_gain) + 1

    return best_feature_index
"""
# ----------------------------------------------------------------------------------------


best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)


# Not graded
tree = []


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree.
        current_depth (int):    Current depth. Parameter used during recursive call.

    """

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-" * current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)
# ----------------------------------------------------------------------------------------


build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
utils.generate_tree_viz(root_indices, y_train, tree)
