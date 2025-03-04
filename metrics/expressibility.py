import numpy as np
from scipy.special import kl_div

import numpy as np


def compute_fidelity_matrix_optimized(dist_list):
    """
    Compute the fidelity matrix between all pairs of distributions using a vectorized approach.

    Args:
        dist_list (list): A list of distributions, each represented as a dictionary.

    Returns:
        np.ndarray: An upper triangular matrix with ones on the diagonal and fidelities elsewhere.
    """
    # Collect all unique bitstrings
    unique_bitstrings = sorted(set().union(*dist_list))

    # Convert distributions to a matrix (each row is a probability distribution)
    prob_matrix = np.array([
        [dist.get(bit, 0) for bit in unique_bitstrings] for dist in dist_list
    ])

    # Compute Bhattacharyya coefficient:

    # Compute square root of probabilities
    sqrt_probs = np.sqrt(prob_matrix)

    # Compute Bhattacharyya coefficients using matrix multiplication
    # Equivalent to summing sqrt(P*Q) for all pairs
    # Then square the Bhattacharyya coefficient to get fidelity
    fidelity_matrix = (sqrt_probs @ sqrt_probs.T)**2

    # Diagonals should be 1
    np.fill_diagonal(fidelity_matrix, 1)

    return fidelity_matrix.ravel()


def compute_fidelity_w_shots(dists0, dists1, shots=25_000):
    """
    Calculate the fidelity between two distributions using the Bhattacharyya coefficient.

    Parameters:
    dists0 (dict): First distribution with bitstrings as keys and counts as values.
    dists1 (dict): Second distribution with bitstrings as keys and counts as values.
    shots (int): Number of shots (default is 25,000).

    Returns:
    float: The calculated fidelity.
    """
    fidelity = 0.0

    # Normalize counts to probabilities and compute the Bhattacharyya coefficient
    for bitstring in set(dists0.keys()).union(set(dists1.keys())):
        # Normalize counts to probabilities
        p = dists0.get(bitstring, 0) / shots
        # Normalize counts to probabilities
        q = dists1.get(bitstring, 0) / shots
        fidelity += np.sqrt(p * q)

    # Fidelity is the square of the Bhattacharyya coefficient
    return fidelity ** 2


def compute_fidelities_w_shots(dist_list, shots=25_000):
    '''
    Compute the fidelities between all pairs of distributions.
    Args:
        dist_list (list): A list of distributions, each represented as a dictionary.
        shots (int): Number of shots (default is 25,000).
    Returns:
        list: A list of fidelities between all pairs of distributions.
    '''
    fidelities = []
    for i, dist0 in enumerate(dist_list):
        for j, dist1 in enumerate(dist_list):
            if i < j:
                fidelities.append(
                    compute_fidelity_w_shots(dist0, dist1, shots))
            else:
                fidelities.append(1)

    return np.array(fidelities)


def compute_fidelity(dists0, dists1):
    """
    Calculate the fidelity between two distributions using the Bhattacharyya coefficient.

    Parameters:
    dists0 (dict): First distribution with bitstrings as keys and probs as values.
    dists1 (dict): Second distribution with bitstrings as keys and probs as values.

    Returns:
    float: The calculated fidelity.
    """
    fidelity = 0.0

    # Get probabilities and compute the Bhattacharyya coefficient
    for bitstring in set(dists0.keys()).union(set(dists1.keys())):
        p = dists0.get(bitstring, 0)
        q = dists1.get(bitstring, 0)
        fidelity += np.sqrt(p * q)

    # Fidelity is the square of the Bhattacharyya coefficient
    return fidelity ** 2


def compute_fidelities(dist_list):
    '''
    Compute the fidelities between all pairs of distributions.
    Args:
        dist_list (list): A list of distributions, each represented as a dictionary.
    Returns:
        list: A list of fidelities between all pairs of distributions.
    '''
    fidelities = []
    for i, dist0 in enumerate(dist_list):
        for j, dist1 in enumerate(dist_list):
            if i < j:
                fidelities.append(compute_fidelity(dist0, dist1))
            if i == j:
                fidelities.append(1)
            if i > j:
                fidelities.append(fidelities[j*len(dist_list) + i])

    return np.array(fidelities)


def compute_expressibility_w_shots(list_sampler_dist, num_shots=25_000):
    """
    Compute the expressibility of a list of distributions. For this, computes the fidelities between all pairs of distributions
    and the KL divergence between the distribution of fidelities and the Haar distribution.

    Parameters:
    list_sampler_dist (list): A list of distributions, each represented as a dictionary.
    num_shots (int): Number of shots (default is 25,000).

    Returns:
    float: The expressibility of the list of distributions.
    """
    # Compute the fidelities
    fidelities = compute_fidelities_w_shots(list_sampler_dist, num_shots)
    # Compute the KL divergence
    # Use the same ratio as in the paper (2000 / 75) = 26.7
    n_bins = int(len(list_sampler_dist) // 26.7)
    bin_edges = np.linspace(0, 1, n_bins+1)
    hist, _ = np.histogram(fidelities, bins=bin_edges)
    hist = hist / np.sum(hist)
    # Compute the Haar distribution
    fidelity_haar = 1 / n_bins
    haar_in_bins = np.full(n_bins, fidelity_haar)

    # Compute the KL divergence
    kld = kl_div(hist, haar_in_bins).sum()

    return kld


def compute_expressibility(list_sampler_dist, optimized=True):
    """
    Compute the expressibility of a list of distributions. For this, computes the fidelities between all pairs of distributions
    and the KL divergence between the distribution of fidelities and the Haar distribution.

    Parameters:
    list_sampler_dist (list): A list of distributions, each represented as a dictionary.

    Returns:
    float: The expressibility of the list of distributions.
    """
    # Compute the fidelities
    if optimized:
        fidelities = compute_fidelity_matrix_optimized(list_sampler_dist)
    else:
        fidelities = compute_fidelities(list_sampler_dist)
    # Compute the KL divergence
    # Use the same ratio as in the paper (2000 / 75) = 26.66
    n_bins = max(5, int(len(list_sampler_dist) // 26.66))
    bin_edges = np.linspace(0, 1, n_bins+1)
    hist, _ = np.histogram(fidelities, bins=bin_edges)
    hist = hist / np.sum(hist)
    # Compute the Haar distribution
    fidelity_haar = 1 / n_bins
    haar_in_bins = np.full(n_bins, fidelity_haar)

    # Compute the KL divergence
    kld = kl_div(hist, haar_in_bins).sum()

    return kld
