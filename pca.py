import numpy as np
import matplotlib.pyplot as plt
import os


def pca_via_svd(data, n_components):
    # 1. Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # 2. Run SVD on the centered data
    # full_matrices=False is a standard efficiency practice for SVD
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    # 3. Take the top n_components rows of Vt
    principal_components = Vt[:n_components, :]

    # 4. Project the centered data onto those components
    projected_data = np.dot(centered_data, principal_components.T)

    return projected_data, principal_components, mean


def plot_pca_reconstruction(X_original, projected, pc, mean):
    """
    Reconstructs 1D projected data back to 2D and plots it against the original data.
    """
    # 1. Do the reconstruction math inside the function
    reconstructed = np.dot(projected, pc) + mean

    # 2. Plotting
    plt.figure(figsize=(8, 6))

    # Original data
    plt.scatter(X_original[:, 0], X_original[:, 1], alpha=0.4, label="Original Data")

    # Projected data (mapped back to 2D)
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], color='red', alpha=0.7, label="Projected Data")

    # Principal direction line
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], color='black', linestyle='--', label="Principal Direction")

    plt.axis('equal')
    plt.legend()
    plt.title("PCA Projection")
    plt.savefig('plots/pca.png', bbox_inches='tight')
    plt.show()