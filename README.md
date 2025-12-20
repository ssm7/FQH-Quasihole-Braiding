# Localized-FQH-Quasihole-Framework
**Author: Skyler Morris <br>**
**Last updated: 20 Dec 2025**

## Overview
This repository contains simulation tools for **Luaghlin** aand **Moore-Read** quantum Hall states. <br>
Functionality includes: 
- Hamiltonian construction in various geometries with zero, one or two localized quasiholes 
- Energy spectrum analysis via exact diagonalization 
- Adiabatic processes for leaving the thin-cylinder linmit and quasihole braiding
- Quantum circuit construction for the coherent ground state preperation and braiding under the parent hamiltonian described by Kirmani et al., Phys. Rev. B 108, 064303 (2023)

## Research Usage
**Coherent Laughlin Quasihole State Braiding** <br>
Following the processes outlined in Physical Review B 108, 064303 including:
- Construction of coherent-state wavefunctions from Eq. (4) and Eq.(5) and parent Hamiltonian Eq. (8)
- Adiabatic braiding and berry phase extraction using Eq. (9) and Eq. (10) 
- Wavefunction truncation and quantum circuit creation for berry phase extraction on quantum hardware

**Second Quantized Laughlin Quasihole Braiding**   
Laughlin supporting Hamiltonian:

$$
\hat{H} = \sum_{j=0}^{N_\phi - 1} \sum_{k > |m|}
V_{k,m}\, c^\dagger_{j+m}\ c^\dagger_{j+k}\ c_{j+m+k}c_j
$$

$$
V_{k,m} \propto (k^2 - m^2)\exp \left(-k^2 \frac{k^2 + m^2}{2}\right)
$$

Introducing another flux quantum to the $\nu = \frac{1}{3}$ system and the potential

$$
U_{m,n}(h) = \exp (-\frac{1}{2}(h_y - \kappa m)^2-\tfrac{1}{2}(h_y - \kappa n)^2+ i\, h_x\, \kappa (m-n))
$$

gives the quasihole-supporting Hamiltonian:

$$
\hat{H}_{\text{quasihole}} =
\hat{H} +
\sum_{m,n=0}^{N_\phi - 1}
U_{m,n}(h)\, c^\dagger_n c_m
$$

This localizes the additional flux quantum at position   $h = (h_x, h_y)$.

**Second Quantized Moore-Read Quasihole Braiding** 


