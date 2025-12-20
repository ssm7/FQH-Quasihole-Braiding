'''
Author: Skyler Morris 

waveFunctions.py: 
    Helper module for constructing and adiabatically braiding the projected 
    coherent state quasihole wavefunction for a v = 1/3 FQH system on a cylinder
    as described by Kirmani et al., PHYSICAL REVIEW B 108, 064303 (2023). 
'''

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import matplotlib.pyplot as plt
from copy import deepcopy
import h5py



# Topological section shift from Eq. (4)
def bc_q(q, c):
    return 3 * q + 2 + c

# Single quasihole amplitude from Eq. (4)
def phi_q(h, q, c, kappa):
    hx, hy = h
    return np.exp((-1j * q * (hx * kappa + np.pi)) - (1 / 6) * (hy - kappa * bc_q(q, c)) ** 2)


# Two quasiholes amplitude as in Eq. (5) with the factored approximation
# for seperable quasiholes from Eq. (6)
def phi_q1_q2(h1, h2, q1, q2, t, T, kappa):
    h_minus, h_plus = (h1, h2) if h1[1] < h2[1] else (h2, h1)
    if T <= t < 2*T:  
        t_prime = t - T
    elif 3*T <= t < 4*T:  
        t_prime = t - 3*T
    else:
        t_prime = -1
    if 0 < t_prime <= T/2:
        c_minus = t_prime / T
        c_plus = 1 - (t_prime / T)
    elif T/2 < t_prime <= T:
        c_plus = t_prime / T
        c_minus = 1 - (t_prime / T)
    else:
        c_minus = 0
        c_plus = 1
    phi_1 = phi_q(h_minus, q1, c_minus, kappa)
    phi_2 = phi_q(h_plus, q2, c_plus, kappa)
    return phi_1 * phi_2

# TO DO: Interpolation for intersecting quasiholes
    '''
    magnitude = phi_1 * phi_2
    def alpha(t_prime, T):
        return np.pi * (1 - np.cos(np.pi * t_prime / T)) / 2 
    
    if t_prime != -1:
        alpha_val = alpha(t_prime, T)
        phase = np.exp(1j * ((np.pi - alpha_val) * q1 + alpha_val * q2))
        return magnitude * phase
    else:
        return phi_q(h_minus, q1, 0, kappa) * phi_q(h_plus, q2, 1, kappa)
    '''

# Construct the two qusihole coherent state from Eq. (5)
def construct_wavefunction(N_sites, h1, h2, t, T, kappa):
    wavefunction = np.array([
        phi_q1_q2(h1, h2, q1, q2, t, T, kappa)
        for q1 in range(N_sites) for q2 in range(q1 + 1, N_sites)
    ], dtype=complex)
    return wavefunction / np.linalg.norm(wavefunction)

# Constuct parent hamiltonian from Eq. (8) 
# (projection onto orthogonal complement of the state)
def compute_parent_hamiltonian(wavefunction):
    dim = len(wavefunction)
    H = np.eye(dim) - np.outer(wavefunction, np.conj(wavefunction))
    return H

# Moving quasihole trajectory, tracing path in Fig. 1
def h1_position(t, T, Lx, yt, y2, yb):
    if t < T:
        return np.array([Lx * (1 - t / T), yt])
    elif t < 1.5 * T:
        return np.array([0, yt - 2 * (yt - y2) * (t - T) / T])
    elif t < 2 * T:
        return np.array([0, y2 - (y2 - yb) * (2 * t - 3 * T) / T])
    elif t < 3 * T:
        return np.array([Lx * (t - 2 * T) / T, yb])
    elif t < 3.5 * T:
        return np.array([Lx, yb + 2 * (y2 - yb) * (t - 3 * T) / T])
    elif t <= 4 * T + 1e-10:
        return np.array([Lx, y2 + (yt - y2) * (2 * t - 7 * T) / T])
    else:
        print(t)
        raise ValueError("t is outiside range [0,4T]" )

# Implements T.D.S.E for adiabatic evolutoin from Eq. (9) 
def time_dependent_schrodinger(t, psi, T, Lx, yt, y2, yb, N_sites, kappa):
    h1 = h1_position(t, T, Lx, yt, y2, yb)
    h2 = np.array([Lx / 2, y2])
    wavefunction = construct_wavefunction(N_sites, h1, h2, t, T, kappa)
    H = compute_parent_hamiltonian(wavefunction)
    return -1j * np.dot(H, psi)


def create_truncated_wavefunction(h1, h2, T, kappa, q1_min, q2_star, truncation_threshold=1e-6):
    amplitudes = []
    q1_indices = []

    for q1 in range(q1_min, q2_star):
        amp = phi_q1_q2(h1, h2, q1, q2_star, 2*T, T, kappa)
        amplitudes.append(amp)
        q1_indices.append(q1)

    amplitudes /= np.linalg.norm(amplitudes)
    return amplitudes
    
# Solve Eq. (9) using solve IVP with t in [0, 4T] 
def integrate_schrodinger(T, Lx, yt, y2, yb, N_sites, initial_state, kappa, save_file="schrodinger_data.h5"):
    t_eval = np.linspace(0, 4 * T, 4000)
    sol = solve_ivp(
        time_dependent_schrodinger,
        (0, 4 * T),
        initial_state,
        t_eval=t_eval,
        args=(T, Lx, yt, y2, yb, N_sites, kappa),
        method="RK45",  
    )
    with h5py.File(save_file, "w") as hf:
        hf.create_dataset("times", data=sol.t)
        hf.create_dataset("states", data=sol.y)

    return {"times": sol.t, "states": sol.y}


def segment_schrodinger(t, psi, H_start, H_end, T_seg):
    tau = t / T_seg  
    H_t = (1 - tau) * H_start + tau * H_end
    return -1j * np.dot(H_t, psi)


def integrate_ode(T, Lx, yt, y2, yb, N_sites, initial_state, kappa, num_segments=400, save_file="correct_schrodinger.h5"):
    tf = 4 * T
    t_points = np.linspace(0, tf, num_segments + 1)  
    dt = (4 * T) / num_segments

    times = []
    states = []

    psi = initial_state.copy()
    current_time = 0

    for i in range(num_segments):
        t0 = t_points[i]
        t1 = t_points[i+1]

        h1_start = h1_position(t0, T, Lx, yt, y2, yb)
        h1_end = h1_position(t1, T, Lx, yt, y2, yb)
        h2 = np.array([Lx/2, y2])

        wf_start = construct_wavefunction(N_sites, h1_start, h2, t0, T, kappa)
        wf_end = construct_wavefunction(N_sites, h1_end, h2, t1, T, kappa)

        H_start = compute_parent_hamiltonian(wf_start)
        H_end = compute_parent_hamiltonian(wf_end)

        solver = ode(segment_schrodinger).set_integrator('zvode', method='adams')
        solver.set_initial_value(psi, 0)
        solver.set_f_params(H_start, H_end, dt)

        t_seg = 0
        while solver.successful() and t_seg < dt:
            t_next = min(t_seg + dt, dt)
            psi = solver.integrate(t_next)
            t_seg = t_next
        psi /= np.linalg.norm(psi)

        times.append(t1)
        states.append(psi.copy())

    times = np.array(times)
    states = np.array(states).T

    with h5py.File(save_file, "w") as hf:
        hf.create_dataset("times", data=times)
        hf.create_dataset("states", data=states)

    return {"times": times, "states": states}

def construct_wavefunction_single(N_sites, h, t, T, kappa):
    c = 0  
    wavefunction = np.array([
        phi_q(h, q, c, kappa) for q in range(N_sites)
    ], dtype=complex)
    return wavefunction / np.linalg.norm(wavefunction)

def compute_parent_hamiltonian_single(wavefunction):
    dim = len(wavefunction)
    H = np.eye(dim) - np.outer(wavefunction, np.conj(wavefunction))
    return H
def segment_schrodinger_single(t, psi, H_start, H_end, T_seg):
    tau = t / T_seg
    H_t = (1 - tau) * H_start + tau * H_end
    return -1j * np.dot(H_t, psi)

def integrate_segmented_single_quasihole(T, Lx, yt, y2, yb, N_sites, kappa, num_segments=400, save_file="single_quasihole_segmented.h5"):
    tf = 4 * T
    t_points = np.linspace(0, tf, num_segments + 1)
    dt = (4 * T) / num_segments

    times = []
    states = []

    h0 = h1_position(0, T, Lx, yt, y2, yb)
    psi = construct_wavefunction_single(N_sites, h0, 0, T, kappa)

    current_time = 0

    for i in range(num_segments):
        t0 = t_points[i]
        t1 = t_points[i + 1]

        h_start = h1_position(t0, T, Lx, yt, y2, yb)
        h_end = h1_position(t1, T, Lx, yt, y2, yb)

        wf_start = construct_wavefunction_single(N_sites, h_start, t0, T, kappa)
        wf_end = construct_wavefunction_single(N_sites, h_end, t1, T, kappa)

        H_start = compute_parent_hamiltonian_single(wf_start)
        H_end = compute_parent_hamiltonian_single(wf_end)

        solver = ode(segment_schrodinger_single).set_integrator('zvode', method='adams')
        solver.set_initial_value(psi, 0)
        solver.set_f_params(H_start, H_end, dt)

        t_seg = 0
        while solver.successful() and t_seg < dt:
            t_next = min(t_seg + dt, dt)
            psi = solver.integrate(t_next)
            t_seg = t_next

        psi /= np.linalg.norm(psi)

        times.append(t1)
        states.append(psi.copy())

    times = np.array(times)
    states = np.array(states).T

    with h5py.File(save_file, "w") as hf:
        hf.create_dataset("times", data=times)
        hf.create_dataset("states", data=states)

    return {"times": times, "states": states}

def integrate_C1_or_C2_segment(t_start, t_end, T, Lx, yt, y2, yb, N_sites, psi_init, kappa, save_file=None):
    num_segments = 1000
    t_points = np.linspace(t_start, t_end, num_segments + 1)
    dt = (t_end - t_start) / num_segments

    times = []
    states = []

    psi = deepcopy(psi_init)

    for i in range(num_segments):
        t0 = t_points[i]
        t1 = t_points[i + 1]

        h1_start = h1_position(t0, T, Lx, yt, y2, yb)
        h1_end = h1_position(t1, T, Lx, yt, y2, yb)
        h2 = np.array([Lx / 2, y2])

        wf_start = construct_wavefunction(N_sites, h1_start, h2, t0, T, kappa)
        wf_end = construct_wavefunction(N_sites, h1_end, h2, t1, T, kappa)

        H_start = compute_parent_hamiltonian(wf_start)
        H_end = compute_parent_hamiltonian(wf_end)
    
        solver = ode(segment_schrodinger).set_integrator('zvode', method='adams')
        solver.set_initial_value(psi, 0)
        solver.set_f_params(H_start, H_end, dt)

        t_seg = 0
        while solver.successful() and t_seg < dt:
            t_next = min(t_seg + dt, dt)
            psi = solver.integrate(t_next)
            t_seg = t_next

        psi /= np.linalg.norm(psi)
        times.append(t1)
        states.append(psi.copy())

    times = np.array(times)
    states = np.array(states).T

    if save_file:
        with h5py.File(save_file, "w") as hf:
            hf.create_dataset("times", data=times)
            hf.create_dataset("states", data=states)

    return {"times": times, "states": states, "final_state": psi}

def integrate_C1_or_C2_segment_truncated(t_start, t_end, T, Lx, yt, y2, yb, N_sites,
                                psi_init, kappa, q1_min, q2_star, save_file=None):
    num_segments = 1000
    t_points = np.linspace(t_start, t_end, num_segments + 1)
    dt = (t_end - t_start) / num_segments

    times = []
    states = []

    psi = deepcopy(psi_init)
    dim = len(psi)
    U_seg = np.eye(dim, dtype=complex)  

    for i in range(num_segments):
        t0 = t_points[i]
        t1 = t_points[i + 1]

        h1_start = h1_position(t0, T, Lx, yt, y2, yb)
        h1_end = h1_position(t1, T, Lx, yt, y2, yb)
        h2 = np.array([Lx / 2, y2])

        wf_start = create_truncated_wavefunction(h1_start, h2, T, kappa, q1_min, q2_star, truncation_threshold=1e-2)
        wf_end = create_truncated_wavefunction(h1_end, h2, T, kappa, q1_min, q2_star, truncation_threshold=1e-2)

        H_start = compute_parent_hamiltonian(wf_start)
        H_end = compute_parent_hamiltonian(wf_end)

        solver = ode(segment_schrodinger).set_integrator('zvode', method='adams')
        solver.set_initial_value(psi, 0)
        solver.set_f_params(H_start, H_end, dt)

        t_seg = 0
        while solver.successful() and t_seg < dt:
            t_next = min(t_seg + dt, dt)
            psi = solver.integrate(t_next)
            t_seg = t_next

        psi /= np.linalg.norm(psi)
        times.append(t1)
        states.append(psi.copy())

        Hpr = np.eye(dim) - np.outer(wf_start, wf_start.conj())
        U_step = expm(-1j * Hpr * dt)
        U_seg = U_step @ U_seg

    times = np.array(times)
    states = np.array(states).T

    if save_file:
        with h5py.File(save_file, "w") as hf:
            hf.create_dataset("times", data=times)
            hf.create_dataset("states", data=states)
            hf.create_dataset("U_seg", data=U_seg)

    return {"times": times, "states": states, "final_state": psi, "unitary": U_seg}