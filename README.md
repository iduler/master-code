# Fluid Injection Poromechanics Simulation

A 3D poromechanics model of fluid injection into a fractured subsurface reservoir, built with [PorePy](https://github.com/pmgbergen/porepy).


## Overview

This project simulates the coupled hydraulic and mechanical response of a heterogeneous fractured rock formation subjected to fluid injection. The model is designed to study pressure diffusion, stress redistribution, and fault slip potential.

The setup is inspired by injection-induced seismicity, such as the 2016 Mw 5.1 Fairview earthquake sequence in Oklahoma.

## Key Features

- 3D domain with layered heterogeneous rock properties
- Elliptical fracture with export of slip-related quantities
- Coupled flow and mechanics
- Time-dependent fluid injection on the northern boundary
- Export of results for visualization in ParaView

## Dependencies

- Python 3.10+
- PorePy
- NumPy

## Setup

1. Clone this repo

    ```bash
    git clone https://github.com/iduler/porepy.git
    ```

2. Install PorePy by following this [guide](https://github.com/pmgbergen/porepy).

## How to Run

Do one of these to run the simulation:

- Run in terminal

    ```bash
    python injection_poromechanics_model.py
    ```

- Open `injection_poromechanics_model.py` and press play button in VS Code

## Physics

The model solves fully coupled Biot poromechanics:

- Darcy flow (including fracture permeability)
- Linear elasticity
- Gravity body forces
- Frictional contact mechanics on fractures

## Boundary Conditions

- Top: no-flow (caprock assumption)
- North: strip with time-dependent fluid injection, rest no-flow
- West, East, South, Bottom: hydrostatic pressure
- Mechanics: lithostatic stresses with a roller condition on the northern boundary

## Output

Running `injection_poromechanics_model.py` will produce files in `fluid_injection_3D` that can be opened in ParaView for visualization.

Examples of quantities available for visualization:

- Pressure
- Displacement
- Slip indicator
- Pressure change due to injection
- Displacement change due to injection