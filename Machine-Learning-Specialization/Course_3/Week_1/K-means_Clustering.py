# K-means Clustering
import numpy as np
import matplotlib.pyplot as plt
from Package_1 import utils


# Random initialization - It's a method that randomly associates K centroids 'coordinates' to some X(i) in order
# for the algorithm to start from somewhere.
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X
    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters
    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples. Creates a list with the X.shape[0] dimension and transforms
    # the X(i) values in its index and them creates a np.array with index that are randomly positioned.
    randidx = np.random.permutation(X.shape[0])

    # Uses the first K index from the randidx array and picks those from the X array, resulting in
    # taking K X(i) examples from X array.
    centroids = X[randidx[:K]]

    return centroids
# -------------------------------------------------------------------------------------------------------------


# This function calculates C(i), in other words it calculates de distance between X(i) and the centroid(i) and
# relates every X(i) with an index that belongs to its closest centroid.
# This function is basically the calculation of Cost function 'J()'.
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids
    Returns:
        idx (array_like): (m,) closest centroids
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)

    return idx
# -------------------------------------------------------------------------------------------------------------


# Load an example dataset that we will be using
X = utils.load_data()
print("First five elements of X are:\n", X[:5])
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# Find the closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)
# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])


# This function calculates the mean for each C(i) using its related X(i) and relocates the centroid to be more in
# the center of its cluster.
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of the closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    for k in range(centroids.shape[0]):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)

    return centroids
# -------------------------------------------------------------------------------------------------------------


K = 3
centroids = compute_centroids(X, idx, K)
print("The centroids are:\n", centroids)


# This function uses the two functions above to relocate the centroids x times, depending on the nº of iterations
# we want. We calculate here the Cost function of K-means for a defined number of iterations and find the
# clusters centroid coordinates with the lowest J() found in the number of iterations performed.
def run_kMeans(X, initial_centroids, max_iters=50):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print(f'K-Means iteration {i}/{max_iters-1}')

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx
# -------------------------------------------------------------------------------------------------------------


# Load an example dataset
X = utils.load_data()

# Set initial centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
K = 3

# Number of iterations
max_iters = 100

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
print("The centroids are:\n", centroids)


# Load an image of a bird
original_img = plt.imread('bird_small.png')
# Visualizing the image
plt.imshow(original_img)
print("Shape of original_img is:", original_img.shape)

# Divide by 255 so that all values are in the range 0 - 1
original_img = original_img / 255
# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10
# Using the function you have implemented above.
initial_centroids = kMeans_init_centroids(X_img, K)
# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Represent image in terms of indices
X_recovered = centroids[idx, :]
# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)


# Display original image
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()

# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title(f'Compressed with %d colours{K}')
ax[1].set_axis_off()
