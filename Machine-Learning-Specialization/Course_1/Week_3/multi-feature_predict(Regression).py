import math
import copy
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Applying feature engineering + polynomial regression
X_train2 = np.c_[X_train**2]
X_super_train = np.concatenate((X_train, X_train2), axis=0)
y_super_train = np.concatenate((y_train, y_train**2), axis=0)

# data is stored in numpy array/matrix
print(f'X_train array: \n{X_super_train}')
print(f"X Shape: {X_super_train.shape}, X Type:{type(X_super_train)})\n")
print(f'y_train array: \n{y_super_train}')
print(f"y Shape: {y_super_train.shape}, y Type:{type(y_super_train)})\n")


# -----------------------------------------------------------------------------------------------------------------
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# normalize the original features | --> can use X_train or X_super_train
X_norm, X_mu, X_sigma = zscore_normalize_features(X_super_train)

print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_super_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")


# -----------------------------------------------------------------------------------------------------------------
def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p


# -----------------------------------------------------------------------------------------------------------------
def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter

    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b

    return p


# -----------------------------------------------------------------------------------------------------------------
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


# -----------------------------------------------------------------------------------------------------------------
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


# -----------------------------------------------------------------------------------------------------------------
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    print("Iteration|    Cost    |    w0    |    w1    |    w2    |    w3    |"
          "    b    |    djdw0    |    djdw1    |    djdw2    |    djdw3    |    djdb   ")
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  # None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  # None
        b = b - alpha * dj_db  # None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f" {i:7d}:{J_history[-1]:10.2f} {w[0]:10.2f} {w[1]:10.2f} {w[2]:10.2f} {w[3]:10.2f} {b:10.2f} "
                  f"{dj_dw[0]:12.2f} {dj_dw[1]:13.2f} {dj_dw[2]:13.2f} {dj_dw[3]:13.2f} {dj_db:12.2f} ")

    return w, b, J_history  # return final w,b and J history for graphing
# -----------------------------------------------------------------------------------------------------------------


# initialize parameters
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.
# some gradient descent settings
iterations = 10000
alpha = 8.3e-7

# run gradient descent | --> Can use X_train or X_norm, to use x_norm it's better to have a big data set
w_final, b_final, J_hist = gradient_descent(X_super_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)

print(f"\n--> b = {b_final:0.2f} | w found by gradient descent = {w_final}\n ")

m, _ = X_train.shape
for i in range(m):
    print(f"Prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
