import numpy as np

import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_ansatz_landscape(individuals_parameter, expected_values, method='TSNE', interp_method='linear', mode='tri'):
    '''
    Plot the trainability landscape of the ansatz parameters.

    Parameters:
        individuals_parameter (np.ndarray): 2D array where individuals are rows and parameters are columns
        expected_values (np.ndarray): 1D array of expected values

    Returns:
        None
    '''

    if method == 'TSNE':
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(individuals_parameter)
        plot_landscape(
            X_embedded[:, 0], X_embedded[:, 1], expected_values, interp_method, mode)
    elif method == 'PCA':
        pca = PCA(n_components=2, random_state=42)
        X_embedded = pca.fit_transform(individuals_parameter)
        plot_landscape(
            X_embedded[:, 0], X_embedded[:, 1], expected_values, interp_method, mode)


def plot_landscape(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, interp_method='linear', mode='tri'):
    """
    Plots the optimization landscape based on an array of 3D points.

    Parameters:
        X (np.ndarray): 1D array of X coordinates
        Y (np.ndarray): 1D array of Y coordinates
        Z (np.ndarray): 1D array of Z coordinates

    Returns:
        None
    """
    # Plot the surface of the optimization landscape
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='plasma',
                    edgecolor='none', alpha=0.7)
    ax.set_title('Landscape of Expected Values')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Expected Value')

    # Contour plot to highlight barren plateaus
    ax2 = fig.add_subplot(122)
    if mode == 'tri':
        triang = Triangulation(X, Y)
        contour = ax2.tricontourf(triang, Z, levels=50, cmap='plasma')
    elif mode == 'interp':
        grid_size_X = int(np.sqrt(len(X)))
        grid_size_Y = int(np.sqrt(len(Y)))

        # Create a grid to interpolate values
        xi = np.linspace(X.min(), X.max(), grid_size_X)
        yi = np.linspace(Y.min(), Y.max(), grid_size_Y)
        X_grid, Y_grid = np.meshgrid(xi, yi)

        # Interpolating Z values on the grid
        Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method="linear")

        # # Compute gradients to detect barren plateaus
        grad_X, grad_Y = np.gradient(Z_grid)
        grad_magnitude = np.sqrt(grad_X**2 + grad_Y**2)
        contour = ax2.contourf(X_grid, Y_grid, np.log1p(grad_magnitude), levels=50, cmap='plasma')
    plt.colorbar(contour, ax=ax2, label='Expected Value')
    ax2.set_title('Map of Expected Values')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    plt.suptitle('Optimization Landscape and Barren Plateau Detection')
    plt.show()

    return barren_plateau_exists(Z)

def barren_plateau_exists(values: np.array, median_readout_error=1.325e-2, threshold=5e-2):
    """Barren plateau detection for VQE

    Args:
        values (np.array): data to check for barren plateau
        median_readout_error (float): median readout error
        threshold (float): threshold for barren plateau detection

    Returns:
        bool: True if barren plateau, False otherwise
    """

    return np.std(values) < threshold + median_readout_error, np.std(values)
