'''
Author: Skyler Morris 

fullCircuit.py: 
    Helper module for constructing the ground state preperation and brading circuits 
     described by Kirmani et al., PHYSICAL REVIEW B 108, 064303 (2023). 
'''

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import U3Gate
from itertools import product
from tools.coherent.waveFunctions import phi_q1_q2
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.compiler import Compiler
from bqskit.passes import LEAPSynthesisPass

def generate_control_patterns(num_controls):
    patterns = list(product([0, 1], repeat=num_controls))
    return sorted(patterns, key=lambda x: x[::-1])

def create_controlled_u3_circuit(num_qubits):

    qr = QuantumRegister(num_qubits, 'q')
    circuit = QuantumCircuit(qr)
    alpha_curr = 0
    for target_qubit in range(num_qubits):
        control_patterns = generate_control_patterns(target_qubit)  

        for control_state in control_patterns:
            theta = 2 
            phi = 0
            lambd = np.pi
            u3_gate = U3Gate(theta, phi, lambd)
            alpha_curr = alpha_curr + 1

            if target_qubit == 0:
                circuit.append(u3_gate, [qr[target_qubit]])
            else:
                control_indices = [i for i, bit in enumerate(control_state)]

                ctrl_state = int("".join(map(str, control_state)), 2)

                cu3_gate = u3_gate.control(len(control_indices), ctrl_state=ctrl_state)
                circuit.append(cu3_gate, control_indices + [qr[target_qubit]])

    return circuit
def get_gaussian_peak_q(hy, kappa, c=0):
    return round(hy / (3 * kappa) - (2 + c) / 3)

def truncated_wavefunction(h1, h2, T, kappa, truncation_threshold=1e-6, c=0):
    q2_star = get_gaussian_peak_q(h2[1], kappa, c)

    amplitudes = []
    q1_indices = []

    for q1 in range(q2_star):
        amp = phi_q1_q2(h1, h2, q1, q2_star, 2*T, T, kappa)
        amplitudes.append(amp)
        q1_indices.append(q1)

    amplitudes = np.array(amplitudes, dtype=complex)
    amplitudes /= np.linalg.norm(amplitudes)

    mask = np.abs(amplitudes) > truncation_threshold
    amplitudes = amplitudes[mask]
    q1_indices = np.array(q1_indices)[mask]

    if amplitudes.size == 0:
        raise ValueError("All amplitudes dropped after truncation.")

    amplitudes /= np.linalg.norm(amplitudes)

    return amplitudes


def get_q1min_q2star(h1, h2, kappa, T, truncation_threshold=1e-6, c=0):
    q2_star = get_gaussian_peak_q(h2[1], kappa, c)
    q1_min = None
    for q1 in range(q2_star):  
        amp = phi_q1_q2(h1, h2, q1, q2_star, 2*T, T, kappa)
        if abs(amp) > truncation_threshold:
            q1_min = q1
            break
    if q1_min is None:
        raise ValueError("No valid q1 found above threshold.")
    return q1_min, q2_star

def create_truncated_wavefunction(h1, h2, T, kappa, q1_min, q2_star, truncation_threshold=1e-6):
    amplitudes = []
    q1_indices = []

    for q1 in range(q1_min, q2_star):
        amp = phi_q1_q2(h1, h2, q1, q2_star, 2*T, T, kappa)
        amplitudes.append(amp)
        q1_indices.append(q1)

    amplitudes /= np.linalg.norm(amplitudes)
    return amplitudes


def compute_alphas(amplitudes):
    a000, a001, a010, a011, a100, a101, a110, a111 = np.abs(amplitudes)
    
    alpha1 = np.arctan(np.sqrt((a100**2 + a101**2 + a110**2 + a111**2) /
                               (a000**2 + a001**2 + a010**2 + a011**2)))


    alpha2_0 = np.arctan(np.sqrt((a010**2 + a011**2) / (a000**2 + a001**2)))
    alpha2_1 = np.arctan(np.sqrt((a110**2 + a111**2) / (a100**2 + a101**2)))

    alpha3_00 = np.arctan(a001 / a000) 
    alpha3_01 = np.arctan(a011 / a010) 
    alpha3_10 = np.arctan(a101 / a100)
    alpha3_11 = np.arctan(a111 / a110) 

    alphas = np.array([alpha1, alpha2_0, alpha2_1, alpha3_00, alpha3_01, alpha3_10, alpha3_11])

    return alphas

def create_controlled_u3_circuit(num_qubits, alphas):

    qr = QuantumRegister(num_qubits, 'q')
    circuit = QuantumCircuit(qr)

    alpha_idx = 0
    for target_qubit in range(num_qubits):
        control_patterns = generate_control_patterns(target_qubit)
        for control_state in control_patterns:
            if alpha_idx >= len(alphas):
                break

            theta = 2 * alphas[alpha_idx]
            phi = 0
            lambd = np.pi
            gate = U3Gate(theta, phi, lambd)

            if target_qubit == 0:
                circuit.append(gate, [qr[target_qubit]])
            else:
                control_indices = [i for i in range(target_qubit)]
                ctrl_state = int("".join(map(str, control_state)), 2)
                controlled_gate = gate.control(len(control_indices), ctrl_state=ctrl_state)
                circuit.append(controlled_gate, control_indices + [qr[target_qubit]])

            alpha_idx += 1

    return circuit

def truncated_wavefunction2(h1, h2, T, kappa, truncation_threshold=1e-6, c=0):
    q2_star = get_gaussian_peak_q(h2[1], kappa, c)

    full_state = np.zeros(2**3, dtype=complex)

    max_q1 = min(q2_star, 2**3)
    for q1 in range(max_q1):
        amp = phi_q1_q2(h1, h2, q1, q2_star, 2*T, T, kappa)
        full_state[q1] = amp

    full_state[np.abs(full_state) < truncation_threshold] = 0

    norm = np.linalg.norm(full_state)
    if norm == 0:
        raise ValueError("All amplitudes dropped after truncation.")
    full_state /= norm

    return full_state

def create_ground_state_prep_3q(alphas):

    q = QuantumRegister(3, 'q')
    qc = QuantumCircuit(q)

    qc.append(U3Gate(2 * alphas[0], 0, np.pi), [q[0]])

    qc.append(U3Gate(2 * alphas[1], 0, np.pi).control(1, ctrl_state=0), [q[0], q[1]])
    qc.append(U3Gate(2 * alphas[2], 0, np.pi).control(1, ctrl_state=1), [q[0], q[1]])

    qc.append(U3Gate(-2 * alphas[3], 0, 0).control(2, ctrl_state=0b00), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[4], 0, 0).control(2, ctrl_state=0b10), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[5], 0, 0).control(2, ctrl_state=0b01), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[6], 0, 0).control(2, ctrl_state=0b11), [q[0], q[1], q[2]])

    return qc

def decompose_unitary (unitary):
    U = UnitaryMatrix(unitary)
    bq_circ = Circuit.from_unitary(U)
    with Compiler() as compiler:
        compiled = compiler.compile(bq_circ, [LEAPSynthesisPass()])
    return(compiled)

def create_ground_state_prep_3q(alphas):

    q = QuantumRegister(3, 'q')
    qc = QuantumCircuit(q)

    qc.append(U3Gate(2 * alphas[0], 0, np.pi), [q[0]])

    qc.append(U3Gate(2 * alphas[1], 0, np.pi).control(1, ctrl_state=0), [q[0], q[1]])
    qc.append(U3Gate(2 * alphas[2], 0, np.pi).control(1, ctrl_state=1), [q[0], q[1]])

    qc.append(U3Gate(-2 * alphas[3], 0, 0).control(2, ctrl_state=0b00), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[4], 0, 0).control(2, ctrl_state=0b10), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[5], 0, 0).control(2, ctrl_state=0b01), [q[0], q[1], q[2]])
    qc.append(U3Gate(-2 * alphas[6], 0, 0).control(2, ctrl_state=0b11), [q[0], q[1], q[2]])

    return qc


#alpha_values = compute_alphas(amplitudes)
num_qubits = 3  

#circuit = create_controlled_u3_circuit(num_qubits, alpha_values)

#for key, value in alpha_values.items():
#    print(f"{key}: {value:.4f}°")


#print(circuit.draw()) 
