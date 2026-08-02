import numpy as np
import matplotlib.pyplot as plt
import os


def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle the edge case where a vector is all zeros
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(X_train, test_point):

    """
    Computes Euclidean distances from one test point to all of X_train.
    """
    # X_train is shape (60, 2), test_point is shape (2,)
    diff = X_train - test_point

    # Calculate distance across the columns (features)
    distances = np.linalg.norm(diff, axis=1)

    return distances


def knn_predict(X_train, y_train, test_point, k=3):
    """
    Predicts the class of a single test point using KNN.
    """
    # 1. Get distances using our helper function
    distances = euclidean_distance(X_train, test_point)  # Shape: (60,)

    # 2. Find the indices of the k smallest distances
    sorted_indices = np.argsort(distances)
    top_k_indices = sorted_indices[:k]

    # 3. Get the labels of those k nearest neighbors
    k_nearest_labels = y_train[top_k_indices]

    # 4. Find the most common label
    vote_counts = np.bincount(k_nearest_labels)
    return np.argmax(vote_counts)


def plot_knn_prediction(X_train, y_train,k,query):
    """
        Plots the decision boundary for KNN and highlights a specific external query point.
        """
    # 1. Find the min and max limits for our grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # 2. Generate the grid (step size of 0.1)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 3. Flatten the grid into a list of [x, y] points
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 4. Predict the class for EVERY point on the grid to build the background
    predictions = []
    for grid_point in grid_points:
        pred = knn_predict(X_train, y_train, grid_point, k=k)
        predictions.append(pred)

    predictions = np.array(predictions)

    # 5. Reshape back to the grid shape
    Z = predictions.reshape(xx.shape)

    # 6. Plot the colored background (contourf) and the original data
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', label='Training Data')

    # 7. Predict and plot the EXTERNAL query point
    query_prediction = knn_predict(X_train, y_train, query, k=k)

    # Plot the query point as a large yellow star so it's easy to see
    plt.scatter(query[0], query[1], c='yellow', marker='*', s=250, edgecolor='black',
                label=f'Query Point (Pred: {query_prediction})')

    plt.title(f"KNN Decision Boundary (k={k})")
    plt.legend(loc='best')
    plt.savefig('plots/knn_predict.png', bbox_inches='tight')
    plt.show()
