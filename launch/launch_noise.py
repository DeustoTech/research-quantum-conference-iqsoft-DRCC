from data.load_dataset import load_dataset
from .prepare_circuit import embedding, generate_random_params, transpile_circuit
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

from qiskit_ibm_runtime.fake_provider import FakeFez

def main():

    qubits = 5
    reps_list = [5]  # TODO: CHANGE ME
    filename = "data\\data.csv"
    num_params = 4_000
    backend = FakeFez()

    print("Generating the circuits")
    circuits = generate_circuits(qubits, reps_list, backend)

    print("Loading the dataframe")
    df = load_dataset(filename)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print("Preparing the circuits")
    dr_circuits = prepare_circuits(circuits, X, num_params)

    results = []

    print("Launching the circuits")

    for r in tqdm.tqdm(range(len(dr_circuits)), desc="Reps", position=0, leave=True):

        data_circuit_array = dr_circuits[r]
        reps = reps_list[r]

        # index data row * num_circs_per_row
        for y_idx in tqdm.tqdm(range(len(y)), desc="Rows", position=1, leave=True):

            parameter_circuit_array = data_circuit_array[y_idx]

            y_instance = y[y_idx]

            losses = {
                'module': [],
                'half': [],
                'num_ones': []
            }

            all_circuit_state_probabilities = []

            # index num_circs_per_row
            for circuit in tqdm.tqdm(parameter_circuit_array, desc="Circ", position=2, leave=False):
                state_probabilities = calculate_exact_probabilities(circuit)
                all_circuit_state_probabilities.append(state_probabilities)
                class_probabilities = interpret_probabilities(
                    state_probabilities, method='module')
                losses['module'].append(compute_loss(
                    y_instance, class_probabilities, function='cross-entropy'))
                class_probabilities = interpret_probabilities(
                    state_probabilities, method='half')
                losses['half'].append(compute_loss(
                    y_instance, class_probabilities, function='cross-entropy'))
                class_probabilities = interpret_probabilities(
                    state_probabilities, method='num_ones')
                losses['num_ones'].append(compute_loss(
                    y_instance, class_probabilities, function='cross-entropy'))

            # Now compute the metrics

            losses['module'] = np.array(losses['module'])
            losses['half'] = np.array(losses['half'])
            losses['num_ones'] = np.array(losses['num_ones'])

            _, trainability_1_std = barren_plateau_exists(
                losses['module'])
            trainability_module = trainability_1_std

            _, trainability_2_std = barren_plateau_exists(
                losses['half'])
            trainability_half = trainability_2_std

            _, trainability_3_std = barren_plateau_exists(
                losses['num_ones'])
            trainability_num_ones = trainability_3_std

            # min and mean loss
            min_loss_module = np.min(losses['module'])
            mean_loss_module = np.mean(losses['module'])

            min_loss_half = np.min(losses['half'])
            mean_loss_half = np.mean(losses['half'])

            min_loss_num_ones = np.min(losses['num_ones'])
            mean_loss_num_ones = np.mean(losses['num_ones'])

            pbar = tqdm.tqdm(total=1, desc="Expr.",
                             position=2, leave=False)

            expressibility = compute_expressibility(
                all_circuit_state_probabilities)
            pbar.update(1)
            pbar.close()

            results.append(
                (reps, y_idx, expressibility,
                    trainability_module, trainability_half, trainability_num_ones,
                    min_loss_module, mean_loss_module,
                    min_loss_half, mean_loss_half,
                    min_loss_num_ones, mean_loss_num_ones))

    result_df = pd.DataFrame(results, columns=[
        'Data Reuploading Reps', 'Row', 'Expressibility', 'Trainability-module', 'Trainability-half', 'Trainability-num_ones', 'Min Loss-module', 'Mean Loss-module', 'Min Loss-half', 'Mean Loss-half', 'Min Loss-num_ones', 'Mean Loss-num_ones'])
    result_df.to_csv("results.csv", index=False)

    return 0


def generate_circuits(num_qubits, reps_list, backend):
    circuits = []
    for reps in reps_list:
        untranspiled_circuit = DataReuploading(n_qubit_data=num_qubits, reps=reps, entanglement='circular', 
                                variational_gates=["ry", "rz"], n_dense_qubits=0, flatten=True, 
                                skip_final_rotation_layer=True, encoding_gate=RXGate)
        transpiled_circuit = transpile_circuit(untranspiled_circuit, backend)
        circuits.append(transpiled_circuit)
    return circuits

def prepare_circuits(circuits_rep, X, num_circs_per_row):

    num_reps = len(circuits_rep)
    n_circ_params = [
        circuits_rep[i].num_parameters - X.shape[1] for i in range(num_reps)]
    num_rows = len(X)

    circuits_emb = []  # indexed (num_reps * num_rows)
    # Prepares the circuits with each row (each circuit same row)
    for i in range(num_reps):
        circuit = circuits_rep[i]
        for j in range(num_rows):  # For each row in the dataframe
            circuits_emb.append(embedding(circuit, X[j]))

    # Generate random parameters for each circuit
    params = [generate_random_params(
        # indexed (num_reps * num_circs_per_row * num_circ_params)
        num_circs_per_row, n_circ_params[i]) for i in range(num_reps)]

    # Assign the parameters to the circuits
    print("Assigning parameters to the circuits")
    # indexed (num_reps * num_rows * num_circs_per_row * num_circ_params)
    final_circ_array = []
    for rep in tqdm.tqdm(range(num_reps), desc="Reps", position=0, leave=True):
        # list of circuits for each repetition of data reuploading circuit, indexed (num_circs_per_row * num_rows)
        dr_circ_array = [[] for row in range(num_rows)]
        for circ in tqdm.tqdm(range(num_circs_per_row), desc="Circuits", position=1, leave=False):
            params_for_this_row = params[rep][circ]
            for row in range(num_rows):
                embedded_circuit = circuits_emb[rep*num_rows + row]
                dr_circ_array[row].append(embedded_circuit.assign_parameters(
                    params_for_this_row, inplace=False))
        final_circ_array.append(dr_circ_array)

    return final_circ_array


if __name__ == "__main__":
    main()
