import math
import copy
import numpy as np
from Data_base import X_train, y_train

# Regularization it's one of the 3 methods to address overfitting.
# Addressing overfitting:
# - Collect more data;
# - Feature selection;
# - Regularization (reduce the size of the parameters, Wj).

# When "feature engineering" +/or "polynomial regression" it's use on the sigmoid function,
# it can become to complex and that will result in an overfitt model.
# That's when "regularization" enters, to stop the current complex function to overfitt the model.

# On this


# -----------------------------------------------------------------------------------------------------------------
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1 / (1 + np.exp(-z))

    return g


# -----------------------------------------------------------------------------------------------------------------
def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b  # (n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)  # scalar
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)  # scalar

    cost = cost / m  # scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j] ** 2)  # scalar
    reg_cost = (lambda_ / (2 * m)) * reg_cost  # scalar

    total_cost = cost + reg_cost  # scalar
    return total_cost  # scalar
# -----------------------------------------------------------------------------------------------------------------


np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)


# -----------------------------------------------------------------------------------------------------------------
def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.0  # scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw
# -----------------------------------------------------------------------------------------------------------------


np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


# -----------------------------------------------------------------------------------------------------------------
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing
# -----------------------------------------------------------------------------------------------------------------


np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2).reshape(-1,) - 0.5)
print(initial_w)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                      compute_cost_logistic_reg, compute_gradient_logistic_reg,
                                      alpha, iterations, 0)
print(f'W = {w}, b = {b}')


# -----------------------------------------------------------------------------------------------------------------
def predict_0_to_1(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        Predicts a float between 0 and 1
    """
    z = np.dot(X, w) + b
    g = 1 / (1 + np.exp(-z))

    return g


# -----------------------------------------------------------------------------------------------------------------
def predict_0_or_1(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
        Predicts a float 0 or 1
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):
        z_wb = 0
        # Loop over each feature
        for j in range(n):
            # Add the corresponding term to z_wb
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij

        # Add bias term
        z_wb += b

        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5
        return p

    # z = np.dot(X, w) + b
    # g = 1 / (1 + np.exp(-z))
    # if X.ndim > 1:
    #    i = 0
    #    for _ in g:
    #        if g[i] >= 0.5:
    #            g[i] = 1
    #        else:
    #            g[i] = 0
    #        i += 1
    # else:
    #    if g >= 0.5:
    #        g = 1
    #    else:
    #        g = 0
# -----------------------------------------------------------------------------------------------------------------


tmp_X = np.array([50, 80])

p = predict_0_to_1(X_train, w, b)

candidate = 1
while candidate <= X_train.ndim:
    if X_train.ndim <= 1:
        print(f"The candidate's nº{candidate} probability of being admitted is {p * 100} % ")
        candidate += 1
    else:
        for result in p:
            print(f"The candidate's nº{candidate} probability of being admitted is {result * 100} % ")
            candidate += 1

p = p.reshape(y_train.shape)
print('\nTrain Accuracy: %f' % (np.mean(p == y_train) * 100), "%")
