# HW5 - Spring Network Beam Simulation

This repository contains a spring network simulation for analyzing the dynamic behavior of a cantilever beam under gravitational loading. The simulation uses a discrete spring-mass system to model beam deflection and compares results with analytical solutions.

## Overview

The simulation models a cantilever beam as a network of interconnected nodes connected by stretching and bending springs. The beam is fixed at one end and subjected to gravitational forces, allowing analysis of dynamic deflection and steady-state behavior.

## How to Run

### Prerequisites

- Python 3.x
- Required packages: `numpy`, `matplotlib`, `imageio`

### Step 1: Initialize the Spring Network

First, generate the spring network data files (nodes, springs, and parameters):

```bash
python initSpringNetwork.py
```

This will create the following files in `springNetworkData/`:
- `nodes.txt` - Node coordinates and masses
- `stretch_springs.txt` - Stretching spring connections and properties
- `bending_springs.txt` - Bending spring (hinge) connections and properties
- `parameters.txt` - Material and geometric parameters

### Step 2: Run the Simulation

Execute the main simulation:

```bash
python Homework5.py
```

This will:
- Load the spring network data
- Run a time-stepping simulation using Newton-Raphson integration
- Save position and velocity data to `plots/output_data_0s-5s.txt`
- Generate visualization plots at regular intervals
- Continue until steady state is reached or maximum time is exceeded

### Step 3: Analyze Results

Plot the deflection results and compare with analytical solution:

```bash
python plotter.py
```

This script:
- Reads the simulation output data
- Calculates average end height over time
- Compares with expected analytical deflection
- Displays error percentage and difference

### Optional: Create Animation

Generate a video from the simulation plots:

```bash
python create_gif.py
```

This creates an MP4 video (`plots/HW5_1.mp4`) from the generated plot images.

## File Descriptions

### Main Scripts

- **`Homework5.py`** - Main simulation script. Loads spring network data, performs time-stepping integration using Newton-Raphson solver, and saves results. Handles boundary conditions (fixed nodes), external forces (gravity), and steady-state detection.

- **`initSpringNetwork.py`** - Initializes the spring network geometry. Creates a mesh of nodes representing the beam, generates stretching springs (along edges and diagonals), and bending springs (hinges) with appropriate stiffness values. Saves all network data to text files.

- **`plotter.py`** - Post-processing script for analyzing simulation results. Reads output data, extracts end node positions, calculates average deflection, and compares with analytical beam theory predictions. Generates plots showing deflection vs. time and expected vs. actual values.

### Solver and Force Computation

- **`solverNewtonRaphson.py`** - Implements the Newton-Raphson implicit integration scheme. Solves the nonlinear system of equations at each time step by iteratively computing forces, Jacobians, and position corrections until convergence.

- **`getExternalForce.py`** - Computes external gravitational forces acting on each node. Applies downward force proportional to node mass.

- **`getForceJacobianImplicit.py`** - (Referenced but not shown) Computes force Jacobians for implicit integration.

### Visualization

- **`plot.py`** - 2D plotting utility for spring networks. Plots node positions and spring connections in x-y plane.

- **`plotShell.py`** - 3D visualization of the shell/beam structure. Creates 3D plots showing node positions and stretching spring connections over time.

- **`plotShell_hinge.py`** - 3D visualization specifically for bending hinges. Shows the hinge geometry and guide nodes used for bending calculations.

- **`create_gif.py`** - Creates an MP4 video animation from saved plot images. Sorts frames by time and combines them into a video file.

### Utilities

- **`steady_state_checker.py`** - Detects when the simulation reaches steady state by monitoring height variations over a specified time window. Returns true when height difference falls below tolerance.

- **`format_number.py`** - Number formatting utilities for displaying values with appropriate precision and scientific notation when needed.

### Data Files

- **`springNetworkData/nodes.txt`** - Node coordinates (x, y, z) and mass for each node in the network.

- **`springNetworkData/stretch_springs.txt`** - Stretching spring definitions: node indices, reference length, and stiffness for each spring.

- **`springNetworkData/bending_springs.txt`** - Bending spring (hinge) definitions: node indices, guide node indices, and bending stiffness.

- **`springNetworkData/parameters.txt`** - Material properties (Young's modulus E, density ρ), geometric parameters (length, width, thickness), mesh parameters (nL_edge, nW_edge), and derived quantities (area, moment of inertia, distributed load q).

### Library Functions

- **`mae263f_functions/`** - Directory containing computational mechanics functions:
  - Energy and force calculations: `gradEs.py`, `gradEb.py`, `gradEt.py`, `gradTheta.py`
  - Hessian computations: `hessEs.py`, `hessEb.py`, `hessTheta.py`
  - Combined gradient/hessian: `gradEs_hessEs.py`, `gradEb_hessEb.py`, `gradEt_hessEt.py`, `gradEb_hessEb_Shell.py`
  - Material directors and reference frames: `computeMaterialDIrectors.py`, `computeReferenceTwist.py`
  - Parallel transport: `computeSpaceParallel.py`, `computeTimeParallel.py`, `parallel_transport.py`
  - Curvature calculations: `computeKappa.py`, `getKappa.py`
  - Utility functions: `crossMat.py`, `rotateAxisAngle.py`, `signedAngle.py`, `set_axes_equal.py`
  - Visualization: `plotrod.py`, `plotrod_simple.py`

## Simulation Parameters

Key parameters can be modified in `initSpringNetwork.py`:

- **Material Properties**: Young's modulus (E), density (ρ)
- **Geometry**: Length, width, thickness
- **Mesh**: Number of nodes along length (nL_edge) and width (nW_edge)

Simulation settings in `Homework5.py`:

- **Time step**: `dt = 0.001` seconds
- **End time**: `end_time = 120` seconds
- **Steady state tolerance**: Based on thickness
- **Fixed nodes**: First 4 nodes (12 DOF) are fixed

## Output

The simulation generates:

- **`plots/output_data_0s-5s.txt`** - CSV file containing time, positions, and velocities at each time step
- **`plots/plot1_t_*.png`** - 3D visualization images at regular intervals
- **`plots/HW5_1.mp4`** - Video animation (if `create_gif.py` is run)

## Expected Results

The simulation should converge to a steady-state deflection that matches the analytical solution for a cantilever beam under uniform distributed load:

```
Expected deflection = (q * L^4) / (8 * E * I)
```

where:
- `q` = distributed load (N/m)
- `L` = beam length (m)
- `E` = Young's modulus (Pa)
- `I` = area moment of inertia (m⁴)
