# 1
import numpy as np
import logging
# %matplotlib widget
from Package_1 import assigment_utils
from Package_1 import public_tests_a1
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)

# 2.1
# Generate some data
X, y, x_ideal, y_ideal = assigment_utils.gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

# split the data using sklearn routine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

# 2.1.1 Plot Train, Test sets
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_test, y_test, color=assigment_utils.dlc["dlblue"], label="test")
ax.legend(loc='upper left')
plt.show()


# 2.2 ----------------------------------------------------------------------------------------------------------------
# GRADED CELL: eval_mse
def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0

    for i in range(m):
        err_i = ((yhat[i] - y[i]) ** 2)
        err += err_i

    err = err / (2 * m)

    return err


# --------------------------------------------------------------------------------------------------------------------


y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
eval_mse(y_hat, y_tmp)

# 2.3 Compare performance on training and test data ------------------------------------------------------------------
# create a model in sklearn, train on training data
degree = 10
lmodel = assigment_utils.lin_model(degree)
lmodel.fit(X_train, y_train)
# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)
# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

# plot predictions over data range
x = np.linspace(0, int(X.max()), 100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1, 1)
assigment_utils.plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

# Generate  data
X, y, x_ideal, y_ideal = assigment_utils.gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

# Split the data using sklearn routine
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

# 3.1 Plot Train, Cross-Validation, Test -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_cv, y_cv, color=assigment_utils.dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test, color=assigment_utils.dlc["dlblue"], label="test")
ax.legend(loc='upper left')
plt.show()

# 3.2 Finding the optimal degree -------------------------------------------------------------------------------------
max_degree = 9
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
x = np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100, max_degree))  # columns are lines to plot

for degree in range(max_degree):
    lmodel = assigment_utils.lin_model(degree + 1)
    lmodel.fit(X_train, y_train)

    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)

    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)

    y_pred[:, degree] = lmodel.predict(x)

optimal_degree = np.argmin(err_cv) + 1

# Let's plot the result:
plt.close("all")
assigment_utils.plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal,
                                   err_train, err_cv, optimal_degree, max_degree)

# 3.3 Tuning Regularization ------------------------------------------------------------------------------------------
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)
err_cv = np.zeros(num_steps)
x = np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100, num_steps))  # columns are lines to plot

for i in range(num_steps):
    lambda_ = lambda_range[i]
    lmodel = assigment_utils.lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)

    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)

    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)

    y_pred[:, i] = lmodel.predict(x)

optimal_reg_idx = np.argmin(err_cv)

plt.close("all")
assigment_utils.plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv,
                                        optimal_reg_idx, lambda_range)

# 3.4 Getting more data: Increasing Training Set Size (m) ------------------------------------------------------------
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree = assigment_utils.tune_m()
assigment_utils.plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)

# 4 Evaluating a Learning Algorithm (Neural Network) -----------------------------------------------------------------
# 4.1 Data Set
# Generate and split data set
X, y, centers, classes, std = assigment_utils.gen_blobs()
# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)
print("X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)
assigment_utils.plt_train_eq_dist(X_train, y_train, classes, X_cv, y_cv, centers, std)


# 4.2 Evaluating categorical model by calculating classification error -----------------------------------------------
# GRADED CELL: eval_cat_err
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    incorrect = 0

    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1

    cerr = incorrect / m

    return cerr


# --------------------------------------------------------------------------------------------------------------------


y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333")
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250")

# 5 Model Complexity -------------------------------------------------------------------------------------------------
# 5.1 Complex model --------------------------------------------------------------------------------------------------
# GRADED CELL: model
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [Dense(120, activation='relu', name="Layer_1"),
     Dense(40, activation='relu', name="Layer_2"),
     Dense(6, activation='linear', name="Activation_layer")], name="Complex")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01))

model.fit(
    X_train, y_train,
    epochs=1000)
model.summary()

# make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(), axis=1)
assigment_utils.plt_nn(model_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Complex Model")

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv, complex model: {cv_cerr_complex:0.3f}")


# 5.1 Simple model -----------------------------------------------------------------------------------------------------
# GRADED CELL: model_s
tf.random.set_seed(1234)
model_s = Sequential(
    [Dense(6, activation="relu", name="Layer_1"),
     Dense(6, activation="linear", name="Activation_layer")], name="Simple")

model_s.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01))

model_s.fit(
    X_train, y_train,
    epochs=1000)
model_s.summary()

# make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(), axis=1)
assigment_utils.plt_nn(model_predict_s, X_train, y_train, classes, X_cv, y_cv, suptitle="Simple Model")

training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, "
      f"{training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}")


# 6 Regularization ---------------------------------------------------------------------------------------------------
# GRADED CELL: model_r
tf.random.set_seed(1234)
model_r = Sequential(
    [Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="Layer_1"),
     Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="Layer_2"),
     Dense(6, activation='linear', name="Activation_layer")], name="Complex")

model_r.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01))

# BEGIN UNIT TEST
model_r.fit(
    X_train, y_train,
    epochs=1000)
model_r.summary()

# make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(), axis=1)
assigment_utils.plt_nn(model_predict_r, X_train, y_train, classes, X_cv, y_cv, suptitle="Regularized")

training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, "
      f"simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv, regularized: {cv_cerr_reg:0.3f}, "
      f"simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}")


# 7 Iterate to find optimal regularization value ---------------------------------------------------------------------
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models = [None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] = Sequential(
        [
            Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation='linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train, y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")


assigment_utils.plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)

# 7.1 Test
assigment_utils.plt_compare(X_test, y_test, classes, model_predict_s, model_predict_r, centers)
