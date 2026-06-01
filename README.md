# Fluid Injection Poromechanics Simulation

A 3D poromechanics model of fluid injection into a fractured subsurface, built with [PorePy](https://github.com/pmgbergen/porepy).


## Overview

This project simulates the coupled hydraulic and mechanical response of a heterogeneous fractured bedrock subjected to fluid injection. The model is designed to study changes in pressure, stress and slip tendency that result from the injection. 

The setup is inspired by injection-induced seismicity, specifically the 2016 Mw 5.1 Fairview earthquake sequence in Oklahoma.

## Key Features

- 3D domain with layered heterogeneous rock properties
- Elliptical fracture
- Coupled flow and mechanics
- Time-dependent fluid injection on the northern boundary
- Export of results for visualization in ParaView

## Required Programs

These programs are required. Python, PorePy and NumPy are already built into the `porepy/dev` Docker image and do not need separate installation.

| Program | Purpose | Download |
|---------|---------|----------|
| **VS Code** | Editor for writing and running the code | https://code.visualstudio.com |
| **Docker Desktop** | Runs the pre-built environment (the "container") that already has PorePy installed | https://www.docker.com/products/docker-desktop |
| **Git** | Clones this project | https://git-scm.com/downloads |
| **ParaView** | Views the simulation results | https://www.paraview.org/download |

## Setup

### Step 1: Install Docker, VS Code and the Dev Containers extension

Install Docker, VS Code and its Dev Containers extension.

### Step 2: Clone the project

In VS Code, open the Command Palette (`Ctrl+Shift+P`) and run:

```
git clone https://github.com/iduler/master-code.git
```

### Step 3: Open the folder in the container

In VS Code, open the `master-code` folder, press `Ctrl+Shift+P` (for running commands) and type:

```
Dev Containers: Open Folder in Container...
```


## How to Run

The entry point is `src/fluid_injection_model.py`. It imports the other modules in `src/`, so it must be run from inside that directory.

Run eter way:
- **Play button:** open `src/fluid_injection_model.py` and click the ▶ button in the top-right corner of the editor.
- **Terminal:** open a terminal in VS Code (**Terminal → New Terminal**) and run:

    ```bash
    cd src
    python fluid_injection_model.py
    ```

## Physics

- Darcy flow (including fracture permeability)
- Linear poroelasticity
- Gravity body forces
- Frictional contact mechanics on fractures

## Boundary Conditions

- North: strip with time-dependent fluid injection, rest no-flow
- West, East, South, Bottom, Top: hydrostatic pressure
- Mechanics: lithostatic stresses with a roller condition on the northern boundary

## Output

Running `fluid_injection_model.py` writes the results to `master-code/fluid_injection_baseline`. To visualize them, open `data.pvd` from that folder in **ParaView**.

Examples of quantities available for visualization:

- Pressure
- Displacement
- Slip tendency
- Pressure change due to injection
- Displacement change due to injection
- Mean stress change due to injection
- Slip tendency change due to injection
