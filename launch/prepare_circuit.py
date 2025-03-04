import numpy as np
from qiskit.transpiler import generate_preset_pass_manager

def embedding(circuit, data):
    '''
    Embed the data into the circuit using 2*actan(data) angles

    Args:
        circuit: Circuit
        data: Data to embed, numpy array (range unknown)

    Returns:
        Circuit: Circuit with data embedded (range of angles [-pi , pi))
    '''

    # Convert the data from initial to range -pi to pi
    for i in range(len(data)):
        data[i] = 2*np.arctan(data[i])

    return circuit.assign_parameters(
        {f"x[{i}]": data[i] for i in range(len(data))},
        inplace=False)


def generate_random_params(num_circuits, num_params):
    '''
    Generate random parameters for the circuit (range -pi to pi)

    Args:
        num_params: Number of parameters
        num_circuits: Number of circuits

    Returns:
        numpy array: Random parameters
    '''
    # A list of num circuits, each with num params random parameters, and a fixed seed
    np.random.seed(42)
    return np.random.uniform(-np.pi, np.pi, (num_circuits, num_params))


def transpile_circuit(circuit, backend):
    '''
    Transpile the circuit for the backend

    Args:
        circuit: Circuit
        backend: Backend

    Returns:
        Circuit: Transpiled circuit
    '''
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    tr_qc = pm.run(circuit)

    return tr_qc
