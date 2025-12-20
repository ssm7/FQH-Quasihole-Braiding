'''
Author: Skyler Morris 

MooreReadImports.py: 
    Helper module for constructing and manipulating the second quantized 
    Hamiltonian of a v = 5/2 FQH system. Equations are in reference to 
    Voinea et al., Phys. Rev. Research 6, 013105 (2024).
'''
import numpy as np
from itertools import combinations
from openfermion import FermionOperator, normal_ordered, hermitian_conjugated
from openfermion.linalg import jw_configuration_state
from scipy.sparse import lil_matrix

def mr_matrix_element(j1, j2, j3, j4, j5, j6, kappa, norb): 
    if (j1 + j2 + j3) != (j4 + j5 + j6):
        return 0.0
    A_create = (j1 - j2)*(j1 - j3)*(j2 - j3)
    A_annih  = (j6 - j5)*(j6 - j4)*(j5 - j4)
    js = [j1, j2, j3, j4, j5, j6]
    sum_j = sum(js)
    sum_j2 = sum(j**2 for j in js)
    exponent = -0.5 * kappa**2 * (sum_j2 - (sum_j**2)/6)
    return A_create * A_annih * np.exp(exponent)

def dist(a, b, norb):
    return min(abs(a - b), norb - abs(a - b))

def mr_matrix_element_trunc_torus(j1, j2, j3, j4, j5, j6, kappa, norb):
    j1, j2, j3, j4, j5, j6 = [j % norb for j in [j1,j2,j3,j4,j5,j6]]
    if ((j1+j2+j3) % norb) != ((j4+j5+j6) % norb):
        return 0.0
    A_create = (dist(j1,j2,norb) *
                dist(j1,j3,norb) *
                dist(j2,j3,norb))
    A_annih  = (dist(j6,j5,norb) *
                dist(j6,j4,norb) *
                dist(j5,j4,norb))
    js = [j1, j2, j3, j4, j5, j6]
    sum_j = sum(js)
    sum_j2 = sum(j*j for j in js)
    exponent = -0.5 * kappa**2 * (sum_j2 - (sum_j**2)/6)
    return A_create * A_annih * np.exp(exponent)

def mr_matrix_element_torus(j1, j2, j3, j4, j5, j6, kappa, norb):
    if (j1 + j2 + j3) != (j4 + j5 + j6):
        return 0.0
    j1 = min(j1, norb - j1)
    j2 = min(j2, norb - j2)
    j3 = min(j3, norb - j3)
    j4 = min(j4, norb - j4)
    j5 = min(j5, norb - j5)
    j6 = min(j6, norb - j6)
    create = sorted([j1, j2, j3])
    annih  = sorted([j4, j5, j6])
    j1, j2, j3 = create[2], create[1], create[0]
    j4, j5, j6 = annih[0], annih[1], annih[2]
    A_create = (j1 - j2)*(j1 - j3)*(j2 - j3)
    A_annih  = (j6 - j5)*(j6 - j4)*(j5 - j4)
    js = [j1, j2, j3, j4, j5, j6]
    sum_j = sum(js)
    sum_j2 = sum(j*j for j in js)
    exponent = -0.5 * kappa**2 * (sum_j2 - (sum_j**2)/6)
    return A_create * A_annih * np.exp(exponent)

def _add_diag_triplet(H, indices, coeff, norb):
    i1, i2, i3 = [j % norb for j in indices]
    term = FermionOperator([(i1,1),(i1,0),(i2,1),(i2,0),(i3,1),(i3,0)], coeff)
    H += term

def _add_offdiag_triplet(H, creators, annihils, coeff, norb):
    creators = [(j % norb) for j in creators]
    annihils = [(j % norb) for j in annihils]
    op = FermionOperator([(i,1) for i in creators] +
                         [(i,0) for i in annihils], coeff)
    op = normal_ordered(op)
    H += op + hermitian_conjugated(op)

def build_mr_hamiltonian(norb, kappa):
    H = FermionOperator()
    for j1 in range(norb):
        for j2 in range(norb):
            for j3 in range(norb):
                for j4 in range(norb):
                    for j5 in range(norb):
                        for j6 in range(norb):
                            v = mr_matrix_element(j1,j2,j3,j4,j5,j6,kappa,norb)
                            if v != 0:
                                H += normal_ordered(
                                    FermionOperator([(j1,1),(j2,1),(j3,1),
                                                     (j4,0),(j5,0),(j6,0)], v))
    return H

def build_mr_hamiltonian_torus(norb, kappa):
    H = FermionOperator()
    for j1 in range(norb):
        for j2 in range(norb):
            for j3 in range(norb):
                for j4 in range(norb):
                    for j5 in range(norb):
                        for j6 in range(norb):
                            v = mr_matrix_element_torus(j1,j2,j3,
                                                        j4,j5,j6,
                                                        kappa,norb)
                            if v != 0:
                                H += normal_ordered(
                                    FermionOperator([(j1,1),(j2,1),(j3,1),
                                                     (j4,0),(j5,0),(j6,0)], v))
    return H

def build_mr_hamiltonian_qhPotential(norb, kappa, hx, hy):
    H = build_mr_hamiltonian(norb, kappa)
    for n in range(norb):
        for i in range(norb):
            coeff = 100 * np.exp(
                -0.5*(hx - kappa*n)**2
                -0.5*(hx - kappa*i)**2
                + hy * kappa * 1j * (n - i)
            )
            H += FermionOperator([(i,1),(n,0)], coeff)
    return H


def truncated_torus_base(norb, kappa):
    H = FermionOperator()

    # Term 6: coeff6, (p+1, p, p−1)
    coeff6 = 2**2 * np.exp(-2 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+1, p, p-1], coeff6, norb)

    # Term 7: coeff7, (p+2, p+1, p−1)
    coeff7 = (2**2 * 3**2) * np.exp((-14 * kappa**2) / 3)
    for p in range(norb):
        _add_diag_triplet(H, [p+2, p+1, p-1], coeff7, norb)

    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+1, p], coeff7, norb)

    # Term 10: coeff10, (p+2, p, p−2)
    coeff10 = (2**8) * np.exp(-8 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+2, p, p-2], coeff10, norb)

    return H

def truncated_torus(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 8 (off-diagonal) 
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        creators = [p+2, p-1, p-2]
        annihils  = [p+1, p, p-3]
        _add_offdiag_triplet(H, creators, annihils, coeff8, norb)

    # Term 9 (off-diagonal) 
    coeff9 = -2 * 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        creators = [(p+2)%norb, (p+1)%norb, (p-1)%norb]
        annihils = [(p+3)%norb, (p+0)%norb, (p-2)%norb]
        _add_offdiag_triplet(H, creators, annihils, coeff9, norb)

    # Term 11 (diagonal) 
    coeff11 = 32 * np.exp(-8 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p, p-3], coeff11, norb)

    # Term 12 (diagonal) 
    coeff12 = 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+2, p-2], coeff12, norb)

    # Term 13 (diagonal) 
    coeff13 = -3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+1, p-1], coeff13, norb)

    return H

def truncated_torus_qhPotential(norb, kappa, hx, hy):
    H = truncated_torus_base(norb, kappa)

    # Term 8
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H, [p+2,p-1,p-2], [p+1,p,p-3], coeff8, norb
        )

    for n in range(norb):
        for i in range(norb):
            coeff_qh = (
                100 * np.exp(
                    -0.5*(hx - kappa*n)**2
                    -0.5*(hx - kappa*i)**2
                    + hy * kappa * 1j * (n - i)
                )
            )
            H += FermionOperator([(i,1),(n,0)], coeff_qh)

    return H

def truncated_torus_no12(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 8
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        creators = [p+2,p-1,p-2]
        annihils = [p+1,p,p-3]
        _add_offdiag_triplet(H, creators, annihils, coeff8, norb)

    # Term 9
    coeff9 = -2 * 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H,
            [(p+2)%norb, (p+1)%norb, (p-1)%norb],
            [(p+3)%norb, (p+0)%norb, (p-2)%norb],
            coeff9, norb
        )

    # Term 11
    coeff11 = 32 * np.exp(-8 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p,p-3], coeff11, norb)

    # Term 13
    coeff13 = -3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+1,p-1], coeff13, norb)

    return H

def density_density_torus_diag_with8(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 8
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(H,
            [p+2,p-1,p-2], [p+1,p,p-3], coeff8, norb
        )

    # Term 12
    coeff12 = 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+2,p-2], coeff12, norb)

    # Term 13
    coeff13 = -3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+1,p-1], coeff13, norb)

    return H

def density_density_torus_diag_with9(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 9
    coeff9 = -2 * 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H,
            [(p+2)%norb, (p+1)%norb, (p-1)%norb],
            [(p+3)%norb, (p+0)%norb, (p-2)%norb],
            coeff9, norb
        )

    # Term 12
    coeff12 = 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+2,p-2], coeff12, norb)

    # Term 13
    coeff13 = -3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+1,p-1], coeff13, norb)

    return H

def truncated_torus_diag(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 12
    coeff12 = 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+2,p-2], coeff12, norb)

    # Term 13
    coeff13 = -3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3,p+1,p-1], coeff13, norb)

    return H

def truncated_torus_off_diag(norb, kappa):
    H = truncated_torus_base(norb, kappa)

    # Term 8
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H,
            [p+2,p-1,p-2], [p+1,p,p-3],
            coeff8, norb
        )

    # Term 9
    coeff9 = -2 * 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H,
            [(p+2)%norb,(p+1)%norb,(p-1)%norb],
            [(p+3)%norb,(p+0)%norb,(p-2)%norb],
            coeff9, norb
        )

    return H

def truncated_cylinder_base(norb, kappa):
    H = FermionOperator()

    # Term 6 (cylinder)
    coeff6 = 4 * np.exp(-2*kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+1, p, p-1], coeff6, norb)

    # Term 7 (cylinder)
    coeff7 = (2**2 * 3**2) * np.exp((-14 * kappa**2) / 3)
    for p in range(norb):
        _add_diag_triplet(H, [p+2, p+1, p-1], coeff7, norb)
        _add_diag_triplet(H, [p+3, p+1, p], coeff7, norb)

    return H


def truncated_cylinder_only9(norb, kappa):
    H = truncated_cylinder_base(norb, kappa)

    coeff9 = -2 * 3**4 * np.exp(-6*kappa**2)
    for p in range(norb):
        creators = [(p+2)%norb, (p+1)%norb, (p-1)%norb]
        annihils = [(p+3)%norb, (p+0)%norb, (p-2)%norb]
        _add_offdiag_triplet(H, creators, annihils, coeff9, norb)

    return H

def truncated_cylinder12(norb, kappa):
    H = truncated_cylinder_base(norb, kappa)

    coeff12 = 3**4 * np.exp(-6*kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+2, p-2], coeff12, norb)

    return H

def truncated_cylinder(norb, kappa):
    H = truncated_cylinder_base(norb, kappa)

    coeff9 = -2 * 3**4 * np.exp(-6*kappa**2)
    for p in range(norb):
        creators = [(p+2)%norb, (p+1)%norb, (p-1)%norb]
        annihils = [(p+3)%norb, (p+0)%norb, (p-2)%norb]
        _add_offdiag_triplet(H, creators, annihils, coeff9, norb)

    coeff12 = 3**4 * np.exp(-6*kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+2, p-2], coeff12, norb)

    coeff13 = -3**4 * np.exp(-6*kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+3, p+1, p-1], coeff13, norb)

    return H

def build_truncated_hamiltonian(norb, kappa):
    H = FermionOperator()

    # Term 6 
    coeff6 = 2**2 * np.exp(-2 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+1, p, p-1], coeff6, norb)

    # Term 7 
    coeff7 = (2**2 * 3**2) * np.exp((-14 * kappa**2) / 3)
    for p in range(norb):
        _add_diag_triplet(H, [p+2, p+1, p-1], coeff7, norb)
        _add_diag_triplet(H, [p+3, p+1, p], coeff7, norb)

    # Term 10 
    coeff10 = (2**8) * np.exp(-8 * kappa**2)
    for p in range(norb):
        _add_diag_triplet(H, [p+2, p, p-2], coeff10, norb)

    # Term 8 (off-diagonal) 
    coeff8 = 72 * np.exp(-6 * kappa**2)
    for p in range(norb):
        _add_offdiag_triplet(
            H,
            [p+2,p-1,p-2],
            [p+1,p,p-3],
            coeff8,
            norb
        )

    # Term 9 (off-diagonal)
    coeff9 = -2 * 3**4 * np.exp(-6 * kappa**2)
    for p in range(norb):
        creators = [(p+2)%norb,(p+1)%norb,(p-1)%norb]
        annihils = [(p+3)%norb,(p+0)%norb,(p-2)%norb]
        _add_offdiag_triplet(H, creators, annihils, coeff9, norb)

    return H


def build_frustrationFree_hamiltonian_torus(norb, kappa):
    H = FermionOperator()

    coeff6 = 4 * np.exp(-2*kappa**2)
    coeff7 = (2**2 * 3**2) * np.exp((-14*kappa**2)/3)
    coeff10 = (2**8) * np.exp(-8*kappa**2)

    for p in range(norb):
        _add_diag_triplet(H, [p+1,p,p-1], coeff6, norb)
        _add_diag_triplet(H, [p+2,p+1,p-1], coeff7, norb)
        _add_diag_triplet(H, [p+3,p+1,p], coeff7, norb)
        _add_diag_triplet(H, [p+2,p,p-2], coeff10, norb)

    return H

def openfermion_basis(n_qubits, n_electrons):
    return [sum(1 << i for i in occ)
            for occ in combinations(range(n_qubits), n_electrons)]

def build_MR_Matrix(fullOperator, basis_states, dim, norb):
    matrix = lil_matrix((dim, dim), dtype=np.complex128)
    state_vectors = []
    for state in basis_states:
        bitstring = format(state, f'0{norb}b')
        occ = [j for j, b in enumerate(bitstring) if b == '1']
        state_vectors.append(jw_configuration_state(occ, norb))

    for i, psi_i in enumerate(state_vectors):
        for j, psi_j in enumerate(state_vectors):
            matrix[i, j] = np.vdot(psi_i, fullOperator @ psi_j)
    return matrix

def supporting_configurations(eigenvectors, basis_states, norb,
                              num_eigenvectors=5, threshold=1e-3):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf,
                        precision=5, suppress=True)
    for i in range(num_eigenvectors):
        print(f"\nEigenvector {i} components with amplitude > {threshold}:")
        for idx, amp in enumerate(eigenvectors[:, i]):
            if abs(amp) > threshold:
                bitstring = format(basis_states[idx], f'0{norb}b')
                sign = '+' if amp.imag >= 0 else '-'
                print(f"  {bitstring}: {amp.real:.4f}{sign}{abs(amp.imag):.4f}j")


