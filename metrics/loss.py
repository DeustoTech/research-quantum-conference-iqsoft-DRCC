import numpy as np

# import log loss
from sklearn.metrics import log_loss
from metrics.expressibility import compute_fidelity


def compute_loss(y: int, probs: dict, function='log-loss'):
    """
    Compute the loss between the true values and the predicted values.

    Parameters:
        y (int): True value
        probs (dict): Dictionary of probabilities
        function (str): Loss function to use

    Returns:
        float: The calculated loss
    """
    num_classes = len(probs)

    if function == 'cross-entropy':
        # One hot encoding of the true values
        y_true = np.eye(num_classes)[y]
        y_pred = np.array([probs[i] for i in range(num_classes)])

        return cross_entropy(y_true, y_pred)
    elif function == 'log-loss':
        y_pred = np.array([probs[i] for i in range(num_classes)])
        # print([y], [y_pred])
        return log_loss([y], [y_pred], labels=[0, 1])
    elif function == 'fidelity':
        return fidelity_binary_loss(y, probs)
    else:
        raise ValueError('Function not recognized')


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred))/y_true.shape[0]


def fidelity_binary_loss(y_true: int, state_probs: dict) -> float:
    """
    Compute the fidelity loss between the true state and the predicted state.

    Parameters:
        y_true (int): True value
        state_probs (dict): Dictionary of probabilities for each state, being the key str of the state and the value the probability

        Returns: 1 - fidelity
    """

    # Get the |1...1> state for y_true=1 and |0...0> state for y_true=0
    # Get the length of the state (number of qubits)
    len_bs = len(list(state_probs.keys())[0])
    if y_true == 0:
        true_state = np.zeros(len_bs)
    else:
        true_state = np.ones(len_bs)
    # to str
    true_state = ''.join([str(int(i)) for i in true_state])

    # Construct the distribution
    dist = {
        true_state: 1
    }

    # Compute the fidelity
    return 1 - compute_fidelity(dist, state_probs)


def make_into_histogram(losses, bins_edges):
    # Create histogram
    hist_values, _ = np.histogram(losses, bins=bins_edges)
    return hist_values
