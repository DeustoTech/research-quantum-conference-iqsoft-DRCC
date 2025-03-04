from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

def calculate_exact_probabilities(assigned_circ):
    simulator = AerSimulator(method='statevector')
    c = assigned_circ.copy()
    c.save_statevector()
    result = simulator.run(c).result()
    counts = result.get_counts()
    return counts

def calculate_noisy_probabilities(transpiled_assigned_circ, backend=FakeFez()):
    noisy_simulated_backend = AerSimulator(method='statevector').from_backend(backend)

    c = transpiled_assigned_circ.copy()
    c.save_statevector()
   
    result = noisy_simulated_backend.run(c).result()
    counts = result.get_counts()
    return counts