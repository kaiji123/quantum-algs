# from qiskit import QuantumCircuit, Aer, transpile, assemble
import numpy as np
# # Create a quantum circuit with two qubits
# qc = QuantumCircuit(2, 2)

# # Apply a Hadamard gate to the first qubit
# qc.h(0)

# # Apply a CNOT gate to entangle the qubits
# qc.cx(0, 1)

# # Measure both qubits
# qc.measure(0, 0)
# qc.measure(1, 1)

# # Simulate the quantum circuit on a classical computer
# simulator = Aer.get_backend('qasm_simulator')
# compiled_circuit = transpile(qc, simulator)
# job = assemble(compiled_circuit)
# result = simulator.run(job).result()

# # Get the measurement results
# counts = result.get_counts()
# print(counts)


# {'00': 507, '11': 517}
# This code creates a simple quantum circuit in Qiskit, entangles two qubits, and measures them, simulating the result on a classical computer.

# Remember that quantum programming is complex and requires a deep understanding of quantum mechanics. Starting with simple programs and gradually working your way up is a good approach.
from qiskit.visualization import plot_state_qsphere
from qiskit import QuantumCircuit, Aer, transpile, assemble

# Create a quantum circuit with two qubits
qc = QuantumCircuit(2, 2)

# Apply Hadamard gates to both qubits
qc.h(0)
# qc.h(1)

# Apply a CNOT gate with qubit 1 as control and qubit 0 as target
qc.cx(1, 0)




# Measure both qubits
qc.measure(0, 0)
qc.measure(1, 1)

# Simulate the quantum circuit on a classical computer
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the measurement results
counts = result.get_counts()
print(counts)


# {'00': 509, '01': 515}
# this is because the second bit of 00 is 0 therefore not change and 1 for 11 therefore change the first qubit therefore 01





# Create a quantum circuit with one qubit
qc = QuantumCircuit(2,2)

# Apply a Z gate to the qubit
qc.z(0)

# Measure both qubits
qc.measure(0, 0)
qc.measure(1, 1)
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the measurement results
counts = result.get_counts()
print(counts)




## normal not operation
from qiskit import QuantumCircuit, Aer, transpile, assemble

# Create a quantum circuit with one qubit
qc = QuantumCircuit(2,2)

# Apply an X gate to the qubit to both qubits (which is not)
qc.x(1)
qc.x(0)

# Measure both qubits
qc.measure(0, 0)
qc.measure(1, 1)

# Simulate the quantum circuit on a classical computer
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the measurement results
counts = result.get_counts()
print(counts)


# creating bell states



# Create a quantum circuit with two qubits
qc = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit
qc.h(0)

# Apply a CNOT gate with the first qubit as the control and the second qubit as the target
qc.cx(0, 1)

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
job = assemble(transpile(qc, simulator))
result = simulator.run(job).result()

# Get the final state vector
state_vector = result.get_statevector()

print(state_vector)



# Create a quantum circuit with two qubits
qc = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit
qc.h(0)

# Apply a CNOT gate with the first qubit as the control and the second qubit as the target
qc.cx(0, 1)

# Apply a Z gate to the second qubit
qc.z(1)

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
job = assemble(transpile(qc, simulator))
result = simulator.run(job).result()

# Get the final state vector
state_vector = result.get_statevector()

print(state_vector)




# Create a quantum circuit with two qubits
qc = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit
qc.h(0)

# Apply a CNOT gate with the first qubit as the control and the second qubit as the target
qc.cx(0, 1)

# Apply an X gate to the second qubit
qc.x(1)

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
job = assemble(transpile(qc, simulator))
result = simulator.run(job).result()

# Get the final state vector
state_vector = result.get_statevector()

print(state_vector)

from qiskit import QuantumRegister, ClassicalRegister

def firstBellState():

    q= QuantumRegister(2,'q')
    c = ClassicalRegister(2,'c')
    # Create a quantum circuit with two qubits
    qc = QuantumCircuit(q,c)


    # Apply a Hadamard gate to the first qubit
    qc.h(q[0])

    # Apply a CNOT gate with the first qubit as the control and the second qubit as the target
    qc.cx(q[0], q[1])



    # Simulate the circuit
    simulator = Aer.get_backend('statevector_simulator')
    job = assemble(transpile(qc, simulator))
    result = simulator.run(job).result()

    # Get the final state vector
    state_vector = result.get_statevector()

    return state_vector



from qiskit import QuantumCircuit, transpile, assemble, Aer



# Create a quantum circuit with one qubit
qc = QuantumCircuit(1)

# Apply a Hadamard gate to create a superposition
qc.h(0)
qc.draw()
# Measure the qubit
qc.measure_all()

# Simulate the circuit
# simulator = AerSimulator()
# compiled_circuit = transpile(qc, simulator)
# job = assemble(compiled_circuit)
# result = simulator.run(job).result()

# # Get the measurement counts
# counts = result.get_counts()

# # Plot the measurement results
# print(counts)

# {'0': 495, '1': 529}


def grover():
    from qiskit import QuantumCircuit, Aer, transpile, assemble




    # Define the number of qubits and the target item to search for
    n = 4  # Number of qubits
    target_item = 7  # Item to search for (in binary: '0111')

    # Create a quantum circuit with n qubits and an auxiliary qubit
    grover_circuit = QuantumCircuit(n + 1, n)

    # Apply Hadamard gates to all qubits
    grover_circuit.h(range(n))

    # Initialize the auxiliary qubit in the |1⟩ state
    grover_circuit.x(n)
    grover_circuit.h(n)

    # Define the number of Grover iterations (you can adjust this based on the problem size)
    num_iterations = int(np.pi / 4 * np.sqrt(2**n))

    # Grover's algorithm iterations
    for _ in range(num_iterations):
        # Oracle: Apply a phase flip to the target item
        for j in range(n):
            if (target_item >> j) & 1:
                grover_circuit.x(j)
        grover_circuit.mct(list(range(n)), n)  # Multicontrol Toffoli gate (mct)
        for j in range(n):
            if (target_item >> j) & 1:
                grover_circuit.x(j)
        
        # Amplitude amplification
        grover_circuit.h(range(n))
        grover_circuit.x(range(n))
        grover_circuit.h(n)
        grover_circuit.mct(list(range(n)), n)
        grover_circuit.h(n)
        grover_circuit.x(range(n))
        grover_circuit.h(range(n))

    # Measure the first n qubits
    grover_circuit.measure(range(n), range(n))

    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(grover_circuit, simulator)
    job = assemble(compiled_circuit)
    result = simulator.run(job).result()

    # Get the measurement counts
    counts = result.get_counts()

    # Plot the measurement results
    print(counts)



qc= QuantumCircuit(6,6)
for i in range(6):
    qc.h(i)
    qc.rz(2*np.pi/6,i)





# deutsch jozsa algorithm

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram


# Define the Oracle for the constant function (0)
def constant_oracle(qc, n):
    pass  # A constant function oracle does nothing in Deutsch-Jozsa

# Define the Oracle for the balanced function (1 for all non-zero inputs)
def balanced_oracle(qc, n):
    for qubit in range(n):
        qc.cx(qubit, n)  # Use CNOT gates to flip the last qubit if the corresponding input qubit is 1

# Define the number of qubits and the type of oracle
n = 3  # Number of input qubits
oracle_type = 'balanced'  # 'constant' or 'balanced'

# Create a quantum circuit with n+1 qubits (the extra qubit is used as the output qubit)
dj_circuit = QuantumCircuit(n + 1, n)

# Apply Hadamard gates to all qubits except the output qubit
dj_circuit.h(range(n))

# Initialize the output qubit in the |1⟩ state
dj_circuit.x(n)

# Apply Hadamard gate to the output qubit
dj_circuit.h(n)

# Apply the Oracle based on the type (constant or balanced)
if oracle_type == 'constant':
    constant_oracle(dj_circuit, n)
elif oracle_type == 'balanced':
    balanced_oracle(dj_circuit, n)

# Apply Hadamard gates to all input qubits
dj_circuit.h(range(n))

# Measure the input qubits to determine if the function is constant or balanced
dj_circuit.measure(range(n), range(n))

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
compiled_circuit = transpile(dj_circuit, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the measurement counts
counts = result.get_counts()

# Plot the measurement results
print(counts)




# deutsch algorithm
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import AerSimulator

# Define the quantum oracle for a balanced function (flips the output)
def balanced_oracle(qc):
    qc.cx(0, 1)

# Define the quantum oracle for a constant function (does nothing)
def constant_oracle(qc):
    pass  # A constant function oracle does nothing in the Deutsch algorithm

# Create a quantum circuit with two qubits
dj_circuit = QuantumCircuit(2, 1)

# Prepare the initial state |01>
dj_circuit.x(1)

# Apply a Hadamard gate to both qubits
dj_circuit.h(range(2))

# Apply the quantum oracle (choose 'balanced' or 'constant')
oracle_type = 'balanced'  # Change to 'constant' for a constant function
if oracle_type == 'balanced':
    balanced_oracle(dj_circuit)
elif oracle_type == 'constant':
    constant_oracle(dj_circuit)

# Apply Hadamard gate to the first qubit
dj_circuit.h(0)

# Measure the first qubit
dj_circuit.measure(0, 0)

# Simulate the circuit
simulator = AerSimulator()
compiled_circuit = transpile(dj_circuit, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the measurement counts
counts = result.get_counts()

# Plot the measurement results
plot_histogram(counts)


# quantum fourier transform
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.visualization import plot_histogram
import numpy as np

# Define the number of qubits
n = 3  # You can change this value to the desired number of qubits

# Create a quantum circuit with n qubits
qft_circuit = QuantumCircuit(n)

# Apply the Quantum Fourier Transform
for i in range(n):
    for j in range(i):
        qft_circuit.cu1(2 * np.pi / 2**(i - j), j, i)
    qft_circuit.h(i)

# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')
compiled_circuit = transpile(qft_circuit, simulator)
job = assemble(compiled_circuit)
result = simulator.run(job).result()

# Get the final state vector
state_vector = result.get_statevector()

# Display the state vector (amplitudes)
print("State Vector (Amplitudes):\n", state_vector)

# Plot the probabilities of measuring each state
counts = result.get_counts()
plot_histogram(counts)











# quantum phase estimation
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.visualization import plot_histogram
import numpy as np

# Define the unitary operator U for which we want to estimate the phase
# Example: We'll use a simple unitary operator for illustration (U = Z)
U = np.array([[1, 0], [0, -1]])

# Define the number of qubits for precision and the number of counting qubits
n_precision = 3  # Number of precision qubits
n_counting = 2   # Number of counting qubits (controls the number of iterations)

# Create a quantum circuit with n_precision + n_counting qubits
qpe_circuit = QuantumCircuit(n_precision + n_counting, n_precision)

# Apply Hadamard gates to precision qubits
for qubit in range(n_precision):
    qpe_circuit.h(qubit)

# Apply controlled-U gates for the controlled exponentiation
for counting_qubit in range(n_counting):
    for i in range(2 ** counting_qubit):
        qpe_circuit.cu1(2 * np.pi / 2**(counting_qubit + 1), counting_qubit + n_precision - 1, n_precision)

# Apply the inverse Quantum Fourier Transform (QFT) to the precision qubits
for qubit in range(n_precision // 2):
    qpe_circuit.swap(qubit, n_precision - 1 - qubit)
for j in range(n_precision):
    for m in range(j):
        qpe_circuit.cu1(-np.pi / float(2**(j - m)), m, j)
    qpe_circuit.h(j)

# Measure the precision qubits
qpe_circuit.measure(range(n_precision), range(n_precision))

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qpe_circuit, simulator)
job = assemble(compiled_circuit, shots=1024)
result = simulator.run(job).result()

# Get the measurement counts
counts = result.get_counts()

# Convert the measurement results to decimal and estimate the phase
estimated_phase = 0
for outcome in counts:
    measured_decimal = int(outcome, 2)
    estimated_phase += measured_decimal / (2 ** n_precision)

# Display the estimated phase
print("Estimated Phase:", estimated_phase)

# Plot the measurement results
plot_histogram(counts)




# variational quantum eigensolver
# Import necessary libraries from Qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator

# Define your problem: Hamiltonian (the operator representing the energy)
# Replace this with your own Hamiltonian
hamiltonian = PauliSumOp.from_list([("X", -1.0), ("Z", 2.0), ("Y", 0.5)])

# Define the quantum instance (simulator in this case)
backend = AerSimulator()
quantum_instance = QuantumInstance(backend, shots=1024)

# Create a variational form (ansatz) for your circuit
ansatz = EfficientSU2(3, reps=1)  # Modify based on your problem

# Create a VQE instance with the chosen ansatz and backend
vqe = VQE(ansatz=ansatz,
          optimizer='COBYLA',  # Replace with your preferred classical optimizer
          quantum_instance=quantum_instance)

# Run VQE to find the minimum energy
result = vqe.compute_minimum_eigenvalue(hamiltonian)

# Extract the ground state energy and optimal parameters
ground_state_energy = result.optimal_value
optimal_parameters = result.optimal_point

# Print the results
print(f"Ground State Energy: {ground_state_energy}")
print(f"Optimal Parameters: {optimal_parameters}")



# quantum approximate optimizer
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble, Aer, execute
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp, Z, I
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram

# Define the graph as an adjacency matrix
graph = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [0, 1, 1, 0]])

# Define the cost Hamiltonian (Max-Cut problem)
num_nodes = len(graph)
pauli_list = []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if graph[i, j] == 1:
            weight = 1.0  # Weight for the edge
            pauli_list.append(weight * (Z ^ Z).to_pauli_op())

cost_hamiltonian = PauliSumOp(paulis=pauli_list)

# Define QAOA circuit
def qaoa_circuit(p, gamma, beta):
    qreg = QuantumRegister(num_nodes)
    creg = ClassicalRegister(num_nodes)
    qaoa = QuantumCircuit(qreg, creg)
    
    # Apply a layer of Hadamard gates
    qaoa.h(qreg)
    
    for _ in range(p):
        # Apply the cost Hamiltonian
        qaoa += cost_hamiltonian.to_circuit()
        qaoa.rx(2 * gamma, qreg)
        
        # Apply the mixing Hamiltonian (Driver Hamiltonian)
        for i in range(num_nodes):
            qaoa.h(qreg[i])
            qaoa.u1(-2 * beta, qreg[i])
            qaoa.h(qreg[i])
    
    # Measure the result
    qaoa.measure(qreg, creg)
    
    return qaoa

# Define the objective function to be minimized
def objective_function(params):
    p, gamma, beta = params
    qaoa = qaoa_circuit(p, gamma, beta)
    t_qaoa = transpile(qaoa, backend=Aer.get_backend('qasm_simulator'))
    qaoa_job = assemble(t_qaoa, shots=1000)
    results = execute(qaoa_job, backend=Aer.get_backend('qasm_simulator')).result()
    counts = results.get_counts()
    
    expectation = 0.0
    for bitstring, count in counts.items():
        bitstring_cost = sum([graph[i, j] * (int(bitstring[i]) - int(bitstring[j])) for i in range(num_nodes) for j in range(i + 1, num_nodes)])
        expectation += count * bitstring_cost
    
    return expectation / 1000  # Normalize by the number of shots

# Define and run the classical optimization
optimizer = COBYLA(maxiter=50)
initial_params = [1, 0.5, 0.5]  # Initial parameters for p, gamma, and beta
optimal_params, optimal_value, _ = optimizer.optimize(num_vars=3, objective_function=objective_function, initial_point=initial_params)

# Run the QAOA circuit with the optimal parameters
optimal_qaoa = qaoa_circuit(int(optimal_params[0]), optimal_params[1], optimal_params[2])
t_optimal_qaoa = transpile(optimal_qaoa, backend=Aer.get_backend('qasm_simulator'))
optimal_qaoa_job = assemble(t_optimal_qaoa, shots=1000)
optimal_results = execute(optimal_qaoa_job, backend=Aer.get_backend('qasm_simulator')).result()
optimal_counts = optimal_results.get_counts()

# Plot the results
plot_histogram(optimal_counts)
print(f"Optimal Parameters (p, gamma, beta): {optimal_params}")
print(f"Optimal Value (Max-Cut Objective): {optimal_value}")



# data reuploading
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import CircuitStateFn
from qiskit.visualization import plot_bloch_multivector, plot_histogram

# Define a classical dataset (for example, a binary vector)
data = np.array([0, 1, 0, 1])

# Create a quantum circuit with a quantum register and a classical register
n_qubits = len(data)
qreg = QuantumRegister(n_qubits, name='q')
creg = ClassicalRegister(n_qubits, name='c')
circuit = QuantumCircuit(qreg, creg)

# Encode the classical data into the quantum state
for i, bit in enumerate(data):
    if bit == 1:
        circuit.x(qreg[i])  # Apply X gate for a classical 1

# Apply a sequence of quantum gates (you can customize this for your task)
circuit.h(qreg)  # Apply Hadamard gates as an example

# Reupload the quantum information into the classical registers
circuit.measure(qreg, creg)

# Simulate the circuit and get the results
simulator = Aer.get_backend('qasm_simulator')
t_circuit = transpile(circuit, backend=simulator)
job = execute(t_circuit, backend=simulator, shots=1024)
result = job.result()
counts = result.get_counts()

# Visualize the results
print("Classical Data:", data)
print("Measured Counts:", counts)
plot_histogram(counts)




# data encoding with rgb pictures
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, execute
from qiskit.visualization import plot_histogram

# Load an RGB image (you'll need a library like PIL for image processing)
from PIL import Image
image = Image.open('your_image.jpg')  # Replace with the path to your image

# Convert the image to a numpy array
image_array = np.array(image)

# Encode the RGB pixel values into a quantum state
n_qubits = 3 * image_array.shape[0] * image_array.shape[1]  # Three qubits per pixel
qreg = QuantumRegister(n_qubits, name='q')
creg = ClassicalRegister(n_qubits, name='c')
circuit = QuantumCircuit(qreg, creg)

for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        pixel = image_array[i, j]
        for channel in range(3):
            amplitude = pixel[channel] / 255.0  # Normalize pixel values to [0, 1]
            circuit.u3(2 * np.arcsin(np.sqrt(amplitude)), 0, 0, qreg[channel + 3 * (i * image_array.shape[1] + j)])

# Apply quantum operations here (e.g., image filtering)

# Reupload quantum information to classical data
circuit.measure(qreg, creg)

# Simulate the circuit and get the results
simulator = Aer.get_backend('qasm_simulator')
t_circuit = transpile(circuit, backend=simulator)
job = execute(t_circuit, backend=simulator, shots=1024)
result = job.result()
counts = result.get_counts()

# Visualize the results (convert counts back to pixel values)
reconstructed_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
for bitstring, count in counts.items():
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            pixel = np.zeros(3, dtype=np.uint8)
            for channel in range(3):
                qubit_idx = channel + 3 * (i * image_array.shape[1] + j)
                if bitstring[qubit_idx] == '1':
                    pixel[channel] = 255
            reconstructed_image[i, j] = pixel

# Visualize the reconstructed image
reconstructed_image = Image.fromarray(reconstructed_image)
reconstructed_image.show()
