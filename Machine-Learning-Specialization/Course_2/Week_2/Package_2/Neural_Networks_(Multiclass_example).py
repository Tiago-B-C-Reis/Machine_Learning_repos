import numpy as np
import logging
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
from Package_2 import autils
from Package_2 import lab_utils_softmax
plt.style.use('./deeplearning.mplstyle')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
np.set_printoptions(precision=2)
# %matplotlib widget


# 2 - ReLU Activation ------------------------------------------------------------------------------------------------
autils.plt_act_trio()


# 3 - Softmax Function -----------------------------------------------------------------------------------------------
# GRADED CELL: my_softmax
def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """

    ez = np.exp(z)
    sm = ez / np.sum(ez)

    return sm


z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")
plt.close("all")
lab_utils_softmax.plt_softmax(my_softmax)
# --------------------------------------------------------------------------------------------------------------------

# 4 - Neural Networks
# 4.1 Problem Statement
# 4.2 Dataset
# load dataset
X, y = autils.load_data()
# 4.2.1 View the variables
print('The first element of X is: ', X[0])
print('The first element of y is: ', y[0, 0])
# 4.2.2 Check the dimensions of your variables
print('The last element of y is: ', y[-1, 0])
print('The shape of X is: ' + str(X.shape))
print('The shape of y is: ' + str(y.shape))

# 4.2.3 Visualizing the Data
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell
m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

# fig.tight_layout(pad=0.5)
autils.widgvis(fig)
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)


# 4.3 Model representation -------------------------------------------------------------------------------------------
# 4.4 Tensorflow Model Implementation

# 4.5 Softmax placement
# GRADED CELL: Sequential model
tf.random.set_seed(1234)  # for consistent results
model = Sequential([tf.keras.Input(shape=(400,)),
                    Dense(25, activation='relu', name='L1'),
                    Dense(15, activation='relu', name='L2'),
                    Dense(10, activation='linear', name='L3')
                    ], name="my_model")

model.summary()

[layer1, layer2, layer3] = model.layers

# Examine Weights shapes
W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X, y,
    epochs=100)


autils.plot_loss_tf(history)

# Prediction ---------------------------------------------------------------------------------------------------------
image_of_two = X[1015]
autils.display_digit(image_of_two)

prediction = model.predict(image_of_two.reshape(1,400))  # prediction
print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)
print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

yhat = np.argmax(prediction_p)
print(f"np.argmax(prediction_p): {yhat}")

warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell


m, n = X.shape
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
autils.widgvis(fig)
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1, 400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()


print(f"{autils.display_errors(model,X,y)} errors out of {len(X)} images")
