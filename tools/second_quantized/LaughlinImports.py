'''
Author: Skyler Morris 

LaughlinImports.py: 
    Helper module for constructing and manipulating the second quantized 
    Hamiltonian of a v = 1/3 FQH system. Equations are in reference to 
    Kirmani et al., PHYSICAL REVIEW B 108, 064303 (2023).
'''
import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix
from scipy.linalg import eigh, expm
from openfermion.linalg import get_sparse_operator, jw_configuration_state
from openfermion import FermionOperator, normal_ordered
from tools.coherent.waveFunctions import phi_q, bc_q

# Laughlin coefficient Eq. (2)
def matrix_element(k, m, kappa):
    return (k**2 - m**2) * np.exp(-0.5 * kappa**2 * (k**2 + m**2))

# v = 1/3 Laughlin pseudopotential in second quantizatinon. 
# Returning an OpenFermion FermionOperator (an algabriac sum of creation/anhilation operator strings)
def laughlin_hamiltonian_sq(Nphi, kappa):
    H = FermionOperator()

    for j in range (Nphi):
        for k in range (Nphi):
            for m in range(-k+1, k):
                j_m  = (j + m) % Nphi
                j_k  = (j + k) % Nphi
                j_mk = (j + m + k) % Nphi

                coeff = matrix_element(k, m, kappa)

                term = ((j_m, 1), (j_k, 1), (j_mk, 0), (j, 0))
                H += FermionOperator(term, coeff)

    return H

# Potential localizing quasihole excitation at (hx, hy)
def confining_qh_potential(Nphi, kappa, hx, hy):
    H = FermionOperator()
    for n in range (Nphi):
        for i in range (Nphi):
            coeff = 10 * np.exp((-.5 * (hy - kappa * n)**2) - (.5 * (hy - kappa * i)**2) + (hx * kappa * 1j * (n - i)))
            term = ((i, 1),(n, 0))
            H += FermionOperator(term, coeff)
    return H

# 1/3 filling Laughlin pseudopotential Hamiltonian with 1 additional unfilled orbitial 
# localized at (hx, hy) with added confing potential. 
# Returning an OpenFermion FermionOperator (an algabriac sum of creation/anhilation operator strings)
def laughlin_hamiltonian_qhPotential(Nphi, kappa, hx, hy):
    H = FermionOperator()
    H += laughlin_hamiltonian_sq(Nphi, kappa)
    H += confining_qh_potential(Nphi, kappa, hx, hy)
    return H

# 1/3 filling Laughlin pseudopotential Hamiltonian with 2 additional unfilled orbitials 
# localized at (hx, hy) and (h2x, h2y) by added confing potential. 
# Returning an OpenFermion FermionOperator (an algabriac sum of creation/anhilation operator strings)
def laughlin_hamiltonian_q1q2Potentialss(Nphi, kappa, hx, hy, h2x,h2y):
    H = FermionOperator()
    H = laughlin_hamiltonian_sq(Nphi, kappa)
    H += confining_qh_potential(Nphi, kappa, hx, hy)
    H += confining_qh_potential(Nphi, kappa, h2x, h2y)
    return H

# Generate the Fock basis for a fixed particle number. Encodes basis states as integer bitmasks.
def openfermion_basis(n_qubits, n_electrons):
    return [sum(1 << i for i in occ) for occ in combinations(range(n_qubits), n_electrons)]

# Convert a second-quantized FermionOperator into its explicit many-body
# matrix representation over the given Fock basis.
def build_H_Matrix(fullOperator, basis_states, dim, norb):
    matrix = lil_matrix((dim, dim), dtype=np.complex128)
    state_vectors = []
    for state in basis_states:
        bitstring = format(state, f'0{norb}b')
        occ_indices = [j for j, b in enumerate(bitstring) if b == '1']
        state_vec = jw_configuration_state(occ_indices, norb)
        state_vectors.append(state_vec)

    for i, psi_i in enumerate(state_vectors):
        for j, psi_j in enumerate(state_vectors):
            matrix[i, j] = np.vdot(psi_i, fullOperator @ psi_j)
    return matrix

# Print the basis configurations (bitstrings) that have meaningful weight
# in the requested number of eigenvectors. For identifying root patterns and
# dominant configurations of Laughlin/quasihole states.
def supporting_configurations(eigenvectors, basis_states, norb, num_eigenvectors=5, threshold=1e-3):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=5, suppress=True)

    for i in range(num_eigenvectors):
        print(f"\nEigenvector {i} components with amplitude > {threshold}:")
        for idx, amp in enumerate(eigenvectors[:, i]):
            if abs(amp) > threshold:
                bitstring = format(basis_states[idx], f'0{norb}b')[::-1]
                print(f"  {bitstring}: {amp.real:.4f}{'+' if amp.imag >= 0 else '-'}{abs(amp.imag):.4f}j")

# 
def ground_state(Nphi, Nelec, kappa, basis_states, dim):
    H_op = laughlin_hamiltonian_sq(Nphi, kappa)
    H_sparse = get_sparse_operator(H_op, Nphi).tocsc()
    H_dense = build_H_Matrix(H_sparse, basis_states, dim, Nphi).toarray()

    vals, vecs = eigh(H_dense) 
    E0 = vals[:]
    psi0 = vecs[:, :]

    return E0, psi0
#
def quasihole_ground_state(Nphi, Nelec, kappa, basis_states, dim, Ly, hx, hy):
    H_op = laughlin_hamiltonian_qhPotential(Nphi, kappa, Ly, hx, hy)
    H_sparse = get_sparse_operator(H_op, Nphi).tocsc()
    H_dense = build_H_Matrix(H_sparse, basis_states, dim, Nphi).toarray()

    vals, vecs = eigh(H_dense) 
    E0 = vals[:]
    psi0 = vecs[:, :]

    return E0, psi0

def pattern_cq(q, c, Nphi, Nelec):
    occ_base = [(3 * j + c) % (Nphi - 1) for j in range(Nelec)]
    occ_shifted = [(i if i < 3*(q+1)+c else i+1) for i in occ_base]
    return tuple(sorted(i % Nphi for i in occ_shifted))

def liquid_coherent_state(Nphi, Nelec, c, h, kappa, basis_states, U_total):
    psi_root = np.zeros(len(basis_states), dtype=complex)

    for q in range(Nphi // 3):  
        occ_indices = pattern_cq(c, q, Nphi, Nelec)
        state_int = sum(1 << (Nphi - 1 - i) for i in occ_indices)
        if state_int in basis_states:
            i = basis_states.index(state_int)
            psi_root[i] = phi_q(h, q, c, kappa)

    psi_root /= np.linalg.norm(psi_root)
    psi_full = U_total @ psi_root
    return psi_full / np.linalg.norm(psi_full)

'''
def liquid_coherent_state(Nphi, Nelec, c, h, kappa, basis_states, U_total):
    psi_root = np.zeros(len(basis_states), dtype=complex)

    for q in range(Nphi // 3):
        pattern = pattern_cq(q, c, Nphi, Nelec)
        state_int = sum(1 << (Nphi - 1 - i) for i in pattern)
        if state_int in basis_states:
            i = basis_states.index(state_int)
            psi_root[i] = phi_q(h, q, c, kappa)
    
    psi_root /= np.linalg.norm(psi_root)
    
    psi_full = U_total @ psi_root
    psi_full /= np.linalg.norm(psi_full)
    return psi_full
'''
def decode_state_int(n, Nphi):
    return tuple(i for i in range(Nphi) if (n >> i) & 1)

def reverse_bits(n, Nphi):
    rev = 0
    for i in range(Nphi):
        if (n >> i) & 1:
            rev |= 1 << (Nphi - 1 - i)
    return rev

def show_basis_support(vec, basis_states, Nphi, threshold=1e-3):
    for i, amp in enumerate(vec):
        if abs(amp) > threshold:
            bitstring = format(basis_states[i], f'0{Nphi}b')[::-1]
            occ = tuple(j for j, b in enumerate(bitstring) if b == '1')
            print(
                f"bitstring={bitstring}  "
                f"amp={amp.real:+.4f}{amp.imag:+.4f}j  "
                f"|amp|={abs(amp):.4f}"
            )