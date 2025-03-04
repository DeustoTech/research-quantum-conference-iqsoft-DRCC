from data.load_dataset import load_dataset
from .prepare_circuit import embedding, generate_random_params
from ansatz.data_reuploading_circuit import DataReuploading
from qiskit.circuit.library import RXGate
from .simulator import calculate_exact_probabilities
import tqdm
import numpy as np
import pandas as pd

from metrics.expressibility import compute_expressibility
from metrics.trainability import barren_plateau_exists
from metrics.trainability import interpret_probabilities
from metrics.trainability import compute_loss


def main():

    reps_list = [1, 2, 3, 4, 5, 10, 20]  # TODO: CHANGE ME
    filename = "data\\data.csv"
    num_params = 4_000

    print("Loading the dataframe")
    df = load_dataset(filename)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print("Preparing the circuits")

    results = []

    print("Launching the circuits")

    for r in tqdm.tqdm(range(len(reps_list)), desc="Reps", position=0, leave=True):

        reps = reps_list[r]

        # index data row * num_circs_per_row
        for y_idx in tqdm.tqdm(range(len(y)), desc="Rows", position=1, leave=True):

            y_instance = y[y_idx]

            losses = []

            # index num_circs_per_row
            for _ in tqdm.tqdm(range(num_params), desc="Circ", position=2, leave=False):
                # Random value between 0 and 1
                random_value = 0.5
                class_probabilities = {0: random_value, 1: 1 - random_value}
                losses.append(compute_loss(
                    y_instance, class_probabilities, function='cross-entropy'))

            # Now compute the metrics
            losses = np.array(losses)

            _, trainability_std = barren_plateau_exists(
                losses)
            trainability = trainability_std

            # min and mean loss
            min_loss = np.min(losses)
            mean_loss = np.mean(losses)

            results.append(
                (reps, y_idx, trainability, mean_loss, min_loss))

    result_df = pd.DataFrame(results, columns=[
        'Data Reuploading Reps', 'Row', 'Trainability', 'Mean Loss', 'Min Loss'])
    result_df.to_csv("results.csv", index=False)

    return 0

if __name__ == "__main__":
    main()
