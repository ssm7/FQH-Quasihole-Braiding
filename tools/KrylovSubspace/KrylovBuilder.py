import numpy as np

def occ_list(bitmask, Nphi):
    return [i for i in range(Nphi) if (bitmask >> i) & 1]
    
def popcount(x):
    return x.bit_count()

def fermion_parity(bitmask, k):
    mask = (1 << k) - 1
    return -1 if popcount(bitmask & mask) % 2 else 1

#C
def apply_annihil(bitmask, k):
    if not (bitmask >> k) & 1:
        return None, 0  

    sign = fermion_parity(bitmask, k)
    new_state = bitmask ^ (1 << k)
    return new_state, sign

#C^dagger
def apply_create(bitmask, k):
    if (bitmask >> k) & 1:
        return None, 0  
    
    sign = fermion_parity(bitmask, k)
    new_state = bitmask | (1 << k)
    return new_state, sign

#Apply C^dagger at sites i and j, and apply C at
# sites k and l
def apply_two_body(bitmask, i, j, k, l):
    total_sign = 1  

    state, s = apply_annihil(bitmask, l)
    if state is None:
        return None, 0
    total_sign *= s

    state, s = apply_annihil(state, k)
    if state is None:
        return None, 0
    total_sign *= s

    state, s = apply_create(state, j)
    if state is None:
        return None, 0
    total_sign *= s

    state, s = apply_create(state, i)
    if state is None:
        return None, 0
    total_sign *= s

    return state, total_sign

#Collect terms from OpenFermion operator
def expand_fermion_op(of_op):
    terms = []

    for term, coeff in of_op.terms.items():
        if len(term) == 2:

            (p, a), (q, b) = term

            if a == 1 and b == 0:
                i = p
                l = q

            elif a == 0 and b == 1:
                i = q
                l = p
                coeff = -coeff   

            else:
                raise ValueError("Invalid operator structure")

            terms.append(("one_body", i, None, None, l, coeff))
            continue

        elif len(term) == 4:


            ops = list(term)

            creators = [op[0] for op in ops if op[1] == 1]
            annihils = [op[0] for op in ops if op[1] == 0]

            if len(creators) != 2 or len(annihils) != 2:
                raise ValueError("Invalid operator count")


            i, j = creators
            k, l = annihils

            if i > j:
                i, j = j, i
                coeff = -coeff   

            if k > l:
                k, l = l, k
                coeff = -coeff

            terms.append(("two_body", i, j, k, l, coeff))
            continue

        else:
            raise ValueError(f"Unexpected operator length {len(term)} in FermionOperator.")

    return terms

#Construct operator in defined basis (used for Krylov generation)
def build_H_krylov_fast(of_op, basis_states, Nphi):
    dim = len(basis_states)
    basis_index = {b: i for i, b in enumerate(basis_states)}

    terms = expand_fermion_op(of_op)

    H = np.zeros((dim, dim), dtype=np.complex128)

    for col_idx, bitmask in enumerate(basis_states):

        for (kind, i, j, k, l, coeff) in terms:

            if kind == "one_body":
                new_state, s1 = apply_annihil(bitmask, l)
                if new_state is None:
                    continue
                new_state, s2 = apply_create(new_state, i)
                if new_state is None:
                    continue
                total_sign = s1 * s2

            else:  
                new_state, total_sign = apply_two_body(bitmask, i, j, k, l)
                if new_state is None:
                    continue

            if new_state in basis_index:
                row_idx = basis_index[new_state]
                H[row_idx, col_idx] += coeff * total_sign

    return H

#Generate degenerate Laughlin CDW 
def generate_root_states(Nelec):
    patterns = {
        0: "100",
        1: "010",
        2: "001"
    }

    all_roots = []

    for sector in range(3):
        base = patterns[sector]

        for slide in range(Nelec): 
            s = ""

            s += "0" * sector

            for i in range(Nelec):
                s += base
                if i == slide:   
                    s += "0"

            all_roots.append(int(s, 2))

    return all_roots

#First order Laughlin squeezing (1001 -> 0110)
def laughlin_squeezes(state, Nphi):
    children = set()

    for i in range(Nphi):
        i0 = i
        i1 = (i + 1) % Nphi
        i2 = (i + 2) % Nphi
        i3 = (i + 3) % Nphi

        if ((state >> i0) & 1 and
            not (state >> i1) & 1 and
            not (state >> i2) & 1 and
            (state >> i3) & 1):

            new = state
            new &= ~(1 << i0)
            new &= ~(1 << i3)
            new |=  (1 << i1)
            new |=  (1 << i2)
            children.add(new)

        if (not (state >> i0) & 1 and
            (state >> i1) & 1 and
            (state >> i2) & 1 and
            not (state >> i3) & 1):

            new = state
            new |=  (1 << i0)
            new |=  (1 << i3)
            new &= ~(1 << i1)
            new &= ~(1 << i2)
            children.add(new)

    return children

#Apply laughlin_squeezes on root states
def squeeze_from_roots(root_states, Nphi):
    basis = set(root_states)
    frontier = set(root_states)

    while frontier:
        new_frontier = set()
        for s in frontier:
            for t in laughlin_squeezes(s, Nphi):
                if t not in basis:
                    basis.add(t)
                    new_frontier.add(t)
        frontier = new_frontier

    return sorted(basis)