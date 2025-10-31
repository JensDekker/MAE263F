# HW3 - Spring Network Beam Simulation

This directory contains code for simulating the dynamic behavior of a beam using a spring network model. The simulation uses both stretching and bending springs to model the beam's response to applied loads.

## Overview

This homework implements a numerical simulation of a beam subject to external forces. The beam is discretized into nodes connected by stretching and bending springs, allowing for dynamic analysis using implicit time integration methods.

## Files Description

### Main Script

#### `Homework3.py`
The main execution script that orchestrates the entire simulation. It:
- Loads the spring network data (nodes, stretching springs, and bending springs) from text files
- Initializes the simulation parameters (time step, total time, applied forces)
- Calculates rest lengths for stretching springs and rest curvatures for bending springs
- Loops over multiple applied forces and performs time integration using Newton-Raphson method
- Compares simulated deflections with expected analytical deflections
- Generates comparison plots showing expected vs simulated deflections, percentage error, and time-dependent max deflection
- Outputs a results summary table with force, expected deflection, simulated deflection, and percentage error

**Key features:**
- Supports multiple applied force magnitudes (default: 500N to 20000N in 500N increments)
- Applies force at x = 0.75 meters along the beam
- Uses fixed boundary conditions (first node fixed in x and y, last node fixed in y)
- Generates plots at specific time steps (0.25s, 0.5s, 0.75s, 1.0s) for force = 2000N
- Calculates and visualizes percentage error between expected and simulated results

### Solver Modules

#### `solverNewtonRaphson.py`
Implements the implicit Newton-Raphson solver for time integration. This solver:
- Uses an implicit time integration scheme with Newton-Raphson iteration
- Solves the nonlinear system of equations iteratively until convergence
- Updates positions and velocities at each time step
- Handles boundary conditions through free DOF selection
- Returns updated positions and velocities for the next time step

**Function:** `solverNewtonRaphson(t_new, x_old, u_old, free_DOF, W, stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch, bending_stiffness_matrix, bending_index_matrix, l_k_bending, m, dt)`

#### `solverExplicit.py`
Implements an explicit time integration solver (alternative to implicit method). This solver:
- Uses explicit time integration (forward Euler-like method)
- Calculates spring forces and external forces
- Updates positions and velocities explicitly
- Generally faster per time step but may require smaller time steps for stability

**Function:** `solverExplicit(x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k)`

### Force and Jacobian Modules

#### `getForceJacobianImplicit.py`
Computes the force vector and Jacobian matrix for the implicit time integration scheme. This module:
- Calculates inertial forces and Jacobian terms
- Accumulates stretching spring forces and their Jacobians
- Accumulates bending spring forces and their Jacobians
- Handles external forces
- Returns the residual force vector and stiffness Jacobian matrix for the Newton-Raphson solver

**Function:** `getForceJacobianImplicit(x_new, x_old, u_old, W, stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch, bending_stiffness_matrix, bending_index_matrix, l_k_bending, m, dt)`

#### `getExternalForce.py`
Calculates and applies external forces to the system. This module:
- Finds the node closest to a specified x-coordinate (default: 0.75 meters)
- Applies a vertical force (negative y-direction) to that node
- Returns the force vector and the index of the node where force is applied

**Function:** `getExternalForce(node_matrix, magnitude)`

### Initialization Module

#### `initSpringNetwork.py`
Initializes and creates the spring network data files. This script:
- Generates node coordinates along a beam of specified length
- Creates stretching springs connecting adjacent nodes
- Creates bending springs connecting every three consecutive nodes
- Distributes mass evenly across nodes
- Saves data to text files in the `springNetworkData/` directory

**Parameters:**
- `N`: Number of nodes (default: 50)
- `L`: Beam length in meters (default: 1.0 m)
- `EA`: Stretching stiffness (N/m)
- `EI`: Bending stiffness (N·m²)
- `m`: Total beam mass (kg)

**Output files:**
- `nodes.txt`: Node coordinates and masses
- `stretch_springs.txt`: Stretching spring connectivity and stiffnesses
- `bending_springs.txt`: Bending spring connectivity and stiffnesses

### Visualization Module

#### `plot.py`
Generates visualization plots of the beam deformation. This module:
- Plots the beam configuration at a given time step
- Shows stretching springs as connected line segments
- Can overlay expected deflection and expected max deflection location
- Saves plots as PNG files with timestamps and force information

**Function:** `plot(x, index_matrix, t, save_plots=True, output_dir='plots', expected_deflection=None, expected_max_deflection_location=None, applied_force=None)`

### Utility Functions

#### `mae263f_functions/`
A package containing various utility functions for spring network calculations:

- **`__init__.py`**: Package initialization file that imports all utility functions
- **`crossMat.py`**: Cross product matrix computation
- **`computekappa.py`**: Computes curvature for bending springs
- **`crossMat.py`**: Matrix operations for cross products
- **`gradEb.py`**: Gradient of bending energy
- **`gradEb_hessEb.py`**: Combined gradient and Hessian of bending energy
- **`gradEs.py`**: Gradient of stretching energy
- **`gradEs_hessEs.py`**: Combined gradient and Hessian of stretching energy
- **`gradEt_hessEt.py`**: Gradient and Hessian of twisting energy
- **`hessEb.py`**: Hessian matrix of bending energy
- **`hessEs.py`**: Hessian matrix of stretching energy
- **`rotateAxisAngle.py`**: Rotation matrix calculations using axis-angle representation

These functions are used throughout the simulation to compute spring forces, energy gradients, and stiffness matrices.

### Data Files

#### `springNetworkData/`
Directory containing the input data files that define the spring network:

- **`nodes.txt`**: Contains node coordinates (x, y) and mass for each node
  - Format: `x_coordinate, y_coordinate, mass`
  - Each line represents one node

- **`stretch_springs.txt`**: Defines stretching springs connecting pairs of nodes
  - Format: `node1_index, node2_index, stiffness`
  - Each line represents one stretching spring

- **`bending_springs.txt`**: Defines bending springs connecting triplets of nodes
  - Format: `node1_index, node2_index, node3_index, stiffness`
  - Each line represents one bending spring where node2 is the middle node

### Output Files

#### `plots/`
Directory containing generated visualization plots:

- **`plot_expected_vs_simulated.png`**: Comparison plot showing expected vs simulated deflections across different applied forces
- **`plot_percentage_error.png`**: Plot showing percentage error between expected and simulated deflections vs applied force
- **`plot_max_deflection.png`**: Plot showing maximum deflection vs time for a specific applied force (2000N)
- **`plot_t_0.25s_F2000N.png`**: Beam configuration snapshot at t=0.25s with 2000N applied force
- **`plot_t_0.50s_F2000N.png`**: Beam configuration snapshot at t=0.50s with 2000N applied force
- **`plot_t_0.75s_F2000N.png`**: Beam configuration snapshot at t=0.75s with 2000N applied force
- **`plot_t_1.00s_F2000N.png`**: Beam configuration snapshot at t=1.00s with 2000N applied force

### Documentation Files

#### `HW3.pdf`
PDF file containing the homework assignment description, problem statement, and requirements.

#### `Template for Preparation of Papers for IEEE Sponsored Conferences.docx`
Word document template (likely for report submission).

## Dependencies

The code requires the following Python packages:
- `numpy`: For numerical computations and array operations
- `matplotlib`: For plotting and visualization
- `typing`: For type hints

## Usage

1. **Initialize the spring network** (if needed):
   ```bash
   python initSpringNetwork.py
   ```
   This will generate the data files in `springNetworkData/` directory.

2. **Run the main simulation**:
   ```bash
   python Homework3.py
   ```
   This will execute the full simulation, generate plots, and output results to the console and `plots/` directory.

## Simulation Parameters

- **Time step**: `dt = 1e-2` seconds
- **Total simulation time**: `maxTime = 1` second
- **Applied forces**: Range from 500N to 20000N (configurable in `Homework3.py`)
- **Force application location**: x = 0.75 meters
- **Boundary conditions**: 
  - First node: fixed in x and y directions
  - Last node: fixed in y direction

## Notes

- The code contains some hardcoded file paths referencing `HW2/` directories - these may need to be updated to `HW3/` if running in isolation
- The simulation uses an implicit Newton-Raphson solver for stable time integration
- Rest lengths for stretching springs and rest curvatures for bending springs are calculated from the initial configuration
- Expected deflections are calculated using analytical beam theory formulas

