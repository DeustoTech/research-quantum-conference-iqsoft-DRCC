import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, RZGate, RXGate, CZGate
from collections.abc import Callable
from qiskit.circuit import Instruction, ParameterVector


class DataReuploading(QuantumCircuit):
    def __init__(
        self,
        n_qubit_data: int = 0,
        encoding_gate: (
            str
            | type
            | Instruction
            | QuantumCircuit
            | None
        ) = None,
        variational_gates: (
            str
            | type
            | Instruction
            | QuantumCircuit
            | list[str | type | Instruction | QuantumCircuit]
            | None
        ) = None,
        entanglement_gate: (
            str
            | type
            | Instruction
            | QuantumCircuit
            | None
        ) = None,
        entanglement: str | list[list[int]] | Callable[[
            int], list[int]] = "reverse_linear",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: QuantumCircuit | None = None,
        flatten: bool | None = None,
        n_dense_qubits: int = 0,
        name: str = "DataReuploading",
    ):
        num_qubits = n_qubit_data + n_dense_qubits

        # Initialize the parent circuit with required number of qubits
        super().__init__(num_qubits, name=name)

        if flatten is None:
            flatten = True

        if encoding_gate is None:
            encoding_gate = RZGate
        # Default gates if none specified
        if variational_gates is None:
            variational_gates = [RXGate]

        if entanglement_gate is None:
            entanglement_gate = CZGate

        # Store the components of the full Data Reuploading Ansatz
        full_circuit = QuantumCircuit(num_qubits)

        inputs = ParameterVector("x", n_qubit_data*reps)

        # Create nested TwoLocal circuits with state re-uploading
        for rep in range(reps):
            # Prepare the data state as a QuantumCircuit
            data_state = QuantumCircuit(num_qubits)
            for i, input in enumerate(inputs[:n_qubit_data]):
                enc = encoding_gate(input)
                data_state.append(enc, [i])
            for i in range(n_dense_qubits):
                # Initialize dense qubits in superposition
                data_state.h(i + n_qubit_data)

            # Define the TwoLocal block for this repetition
            two_local = TwoLocal(
                num_qubits=num_qubits,
                rotation_blocks=variational_gates,
                entanglement_blocks=entanglement_gate,
                entanglement=entanglement,
                reps=1,
                parameter_prefix=f"{parameter_prefix}_{rep}",
                insert_barriers=insert_barriers,
                initial_state=initial_state,
                flatten=True,
                skip_final_rotation_layer=skip_final_rotation_layer,
                skip_unentangled_qubits=skip_unentangled_qubits,
            )

            # Compose the data state and the TwoLocal block
            if flatten:
                self.compose(data_state, inplace=True)
                self.compose(two_local, inplace=True)
            else:
                full_circuit.compose(data_state, inplace=True)
                full_circuit.compose(two_local, inplace=True)

        # Append the full circuit to the DataReuploading circuit
        if not flatten:
            full_circuit_instruction = full_circuit.to_instruction()
            full_circuit_instruction.name = name
            self.append(full_circuit_instruction, range(num_qubits))

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """Parameter bounds."""
        return self.num_parameters * [(-np.pi, np.pi)]
