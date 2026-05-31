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

The default options in each installer are sufficient.

## How It Fits Together

The project files stay on the local machine and run inside a **container**, a ready-made Linux environment that already has PorePy installed. Two pieces connect them:

- The container is started from the **`porepy/dev`** image, which contains PorePy at `/workdir/porepy`.
- Starting the container links the local project folder into it (the `-v` part of the `docker run` command below). The files stay on the local machine and appear inside the container at `/workspaces/master-code`. Edits made in either place are reflected in the other.

The workspace file (`my_devcontainer_workspace.code-workspace`) shows both folders, the project and the PorePy source, side-by-side in VS Code once connected.

## Setup

> The first run downloads the `porepy/dev` image and takes several minutes. This happens only once.

### Step 1: Start Docker Desktop

Open **Docker Desktop** and leave it running in the background. The sign-in prompt can be skipped. Docker is ready when its whale icon appears in the taskbar/menu bar.

> Docker must be running before the steps below, otherwise the container cannot start.

### Step 2: Install the VS Code extensions

1. Open **VS Code**.
2. Open the **Extensions** view (the four-squares icon in the left sidebar, or `Ctrl+Shift+X`).
3. Install these two (both published by Microsoft):
    - **Dev Containers**: provides the "Attach to Running Container" command used in Step 5.
    - **Docker**: shows and manages containers from the sidebar.

### Step 3: Clone the project

**Inside VS Code:**

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Type **Git: Clone**, select it, paste the address below, and press Enter:

    ```
    https://github.com/iduler/master-code.git
    ```

3. Choose a folder to save it in.

> Note the full path to the `master-code` folder; it is needed in the next step.

### Step 4: Start the container

Open a terminal (PowerShell or Command Prompt) and run these two commands.

1. Download the PorePy image (only needed once):

    ```bash
    docker pull porepy/dev
    ```

2. Start a container. Replace `PATH_TO_PROJECT` with the full path to the `master-code` folder from Step 3, and `my-porepy` with any chosen name:

    ```bash
    docker run -dit --name my-porepy -v PATH_TO_PROJECT:/workspaces/master-code porepy/dev
    ```

    Path example: `-v C:\Users\Name\master-code:/workspaces/master-code`

   > The `-v` part links the local project into the container. Without it, the container has PorePy but not the project code.

### Step 5: Connect VS Code to the container

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Type **Dev Containers: Attach to Running Container** and select it.
3. Select the container named in Step 4 (e.g. `my-porepy`). A new VS Code window opens, running inside the container, and a green badge appears in the bottom-left corner.
4. In that window, choose **File → Open Folder** and open `/workspaces/master-code`, or **File → Open Workspace from File** and pick `my_devcontainer_workspace.code-workspace` to also show the PorePy source.

The environment now has Python, PorePy and NumPy ready to use.

## How to Run

The entry point is `src/fluid_injection_model.py`. It imports the other modules in `src/`, so it must be run from inside that directory.

With the container connected (green badge in the bottom-left of VS Code), run it either way:

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