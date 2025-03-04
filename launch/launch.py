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
from interpretation.interpretation_functions import interpret_probabilities
from metrics.loss import compute_loss, make_into_histogram


def main():

    qubits = 5
    reps_list = [1, 2, 3, 4, 5, 10, 20]  # TODO: CHANGE ME
    filename = "data\\old_data.csv"
    num_circs_per_row = 4000

    # Load the bins
    print("Loading the bins")
    bins_crossentropy = np.loadtxt("data\\cross_entropy_bins.csv")
    bins_fidelity = np.loadtxt("data\\fidelity_bins.csv")

    print("Loading the dataframe")
    df = load_dataset(filename)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print("Launching the circuits")

    # Process each rep separately to limit memory usage
    for rep in tqdm.tqdm(reps_list, desc="Reps", position=0, leave=True):

        results = []

        # Generate template circuit
        circuit_template = generate_circuit(qubits, rep)

        # Determine the number of parameters that will be assigned (excluding the data embedding)
        n_circ_params = circuit_template.num_parameters - X.shape[1]

        # Generate the parameters for this repetition
        params = generate_random_params(num_circs_per_row, n_circ_params)

        results = []

        # Process each row (each data instance) individually
        for y_idx in tqdm.tqdm(range(len(y)), desc="Rows", position=1, leave=True):

            y_instance = y[y_idx]

            # Embed the row's data into the circuit template
            embedded_circuit = embedding(circuit_template, X[y_idx])

            losses = {
                'cross_entropy': [],
                'fidelity': [],
            }

            all_circuit_state_probabilities = []

            # For each circuit instance (i.e. each random parameter assignment)
            for i in tqdm.tqdm(range(num_circs_per_row), desc="Circuits", position=2, leave=False):

                # Assign the parameters to a fresh copy of the embedded circuit
                circuit_instance = embedded_circuit.assign_parameters(
                    params[i], inplace=False)

                # Run the simulation
                state_probabilities = calculate_exact_probabilities(
                    circuit_instance)

                all_circuit_state_probabilities.append(state_probabilities)

                # Convert state probabilities to class probabilities
                class_probabilities = interpret_probabilities(
                    state_probabilities, method='module')

                # Compute losses for this instance
                losses['cross_entropy'].append(
                    compute_loss(y_instance, class_probabilities,
                                 function='cross-entropy')
                )
                losses['fidelity'].append(
                    compute_loss(y_instance, state_probabilities,
                                 function='fidelity')
                )

            # Convert loss lists to numpy arrays to compute metrics
            losses['cross_entropy'] = np.array(losses['cross_entropy'])
            losses['fidelity'] = np.array(losses['fidelity'])

            # Compute trainability metrics (e.g., standard deviation of losses)
            trainability_ce = losses['cross_entropy'].std()
            trainability_fidelity = losses['fidelity'].std()

            # Compute loss statistics for cross-entropy
            min_loss_ce = np.min(losses['cross_entropy'])
            max_loss_ce = np.max(losses['cross_entropy'])
            mean_loss_ce = np.mean(losses['cross_entropy'])
            median_loss_ce = np.median(losses['cross_entropy'])
            hist_ce = make_into_histogram(
                losses['cross_entropy'], bins_crossentropy)

            # Compute loss statistics for fidelity
            min_loss_fidelity = np.min(losses['fidelity'])
            max_loss_fidelity = np.max(losses['fidelity'])
            mean_loss_fidelity = np.mean(losses['fidelity'])
            median_loss_fidelity = np.median(losses['fidelity'])
            hist_fidelity = make_into_histogram(
                losses['fidelity'], bins_fidelity)

            # Compute the expressibility metric for the current data row
            expressibility = compute_expressibility(
                all_circuit_state_probabilities)

            results.append((
                rep, y_idx, expressibility,
                trainability_ce, min_loss_ce, max_loss_ce, mean_loss_ce, median_loss_ce, hist_ce,
                trainability_fidelity, min_loss_fidelity, max_loss_fidelity, mean_loss_fidelity, median_loss_fidelity, hist_fidelity
            ))

        # Save the results for this rep to a CSV file
        result_df = pd.DataFrame(results, columns=[
            'Data Reuploading Reps', 'Row', 'Expressibility',
            'Trainability CrossEntropy',
            'Min CrossEntropy', 'Max CrossEntropy', 'Mean CrossEntropy', 'Median CrossEntropy', 'Hist CrossEntropy',
            'Trainability Fidelity',
            'Min Loss Fidelity', 'Max Loss Fidelity', 'Mean Loss Fidelity', 'Median Loss Fidelity', 'Hist Loss Fidelity'
        ])
        result_df.to_csv(f"results_{rep}.csv", index=False)

    return 0


def generate_circuit(num_qubits, rep):
    return DataReuploading(n_qubit_data=num_qubits, reps=rep, entanglement='circular', variational_gates=[
        "ry", "rz"], n_dense_qubits=0, flatten=True, skip_final_rotation_layer=True, encoding_gate=RXGate)


if __name__ == "__main__":
    main()
