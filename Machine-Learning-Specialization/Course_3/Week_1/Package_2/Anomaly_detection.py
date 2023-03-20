# Anomaly detection
import numpy as np
import matplotlib.pyplot as plt
from Package_2 import utils
import math

# Load the dataset
X_train, X_val, y_val = utils.load_data()

# Display the first five elements of X_train
print("The first 5 elements of X_train are:\n", X_train[:5])
# Display the first five elements of X_val
print("The first 5 elements of X_val are\n", X_val[:5])
# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5])
print('The shape of X_train is:', X_train.shape)
print('The shape of X_val is:', X_val.shape)
print('The shape of y_val is: ', y_val.shape, y_val[y_val == 1].shape)

# Create a scatter plot of the data. To change the markers to blue "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')
# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()


# GRADED FUNCTION: estimate_gaussian -----------------------------------------------------------------------
def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset
    Args:
        X (ndarray): (m, n) Data matrix
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    var_sum = 0

    mu = np.mean(X, axis=0)

    for i in range(m):
        var_sum += (X[i] - mu) ** 2
    var = (1 / m) * var_sum

    return mu, var
# -----------------------------------------------------------------------------------------------------------


# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)
print("Mean of each feature:", mu)
print("Variance of each feature:", var)

# Returns the density of the multivariate normal
# at each data point (row) of X_train
p = utils.multivariate_gaussian(X_train, mu, var)
# Plotting code
utils.visualize_fit(X_train, mu, var)


# I have created this function but this file is importing one from 'utils' package.
def multivariate_gaussian(X, mu, var):
    """Calculates the 'Anomaly detection algorithm' for multi-variable X database."""

    m, n = X.shape
    dev = math.sqrt(var)
    p_val = 0

    for i in range(m):
        p_val *= (1/math.sqrt(2*math.pi * dev)) * math.exp(-((X[i] - mu)**2 / (2*var)))

    return p_val
# -----------------------------------------------------------------------------------------------------------


# GRADED FUNCTION: select_threshold. We are using the 'F1 score'.
def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        prediction = (p_val < epsilon)

        tp = sum(y_val == 1)
        fp = sum((prediction == 1) & (y_val == 0))
        fn = sum((prediction == 0) & (y_val == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * (prec * rec)) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
# -----------------------------------------------------------------------------------------------------------


p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)
print(f'Best epsilon found using cross-validation: {epsilon}')
print(f'Best F1 on Cross Validation Set: {F1}')

# Find the outliers in the training set
outliers = p < epsilon
# Visualize the fit
utils.visualize_fit(X_train, mu, var)
# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize=10, markerfacecolor='none', markeredgewidth=2)


# load the dataset
X_train_high, X_val_high, y_val_high = utils.load_data_multi()
print('The shape of X_train_high is:', X_train_high.shape)
print('The shape of X_val_high is:', X_val_high.shape)
print('The shape of y_val_high is: ', y_val_high.shape)

# Apply the same steps to the larger dataset
# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilities for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilities for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print(f'Best epsilon found using cross-validation: {epsilon_high}')
print(f'Best F1 on Cross Validation Set: {F1_high}')
print(f'# Anomalies found: {sum(p_high < epsilon_high)}')
