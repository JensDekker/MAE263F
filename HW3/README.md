# HW3 - Spring Network Beam Simulation

This directory contains code for simulating the dynamic behavior of a beam using a spring network model. The simulation uses both stretching and bending springs to model the beam's response to applied loads.

## Overview

This homework implements a trajectory-tracking control algorithm for a beam simulation. The beam is discretized into nodes connected by stretching and bending springs, allowing for dynamic analysis using implicit time integration methods. The simulation controls the position of the middle node by iteratively adjusting the last two nodes' positions to guide the middle node along a target circular arc trajectory.

## Files Description

### Main Script

#### `Homework3.py`
The main execution script that orchestrates the entire simulation. It:
- Loads the spring network data (nodes, stretching springs, and bending springs) from text files
- Initializes the simulation parameters (time step, total time)
- Calculates rest lengths for stretching springs and rest curvatures for bending springs
- Defines the target trajectory for the middle node using a circular arc path
- Performs iterative control at each time step to adjust the last two nodes' positions, guiding the middle node to follow the target trajectory
- Uses implicit time integration with Newton-Raphson method for stable simulation
- Generates visualization plots at specific time steps and saves trajectory data

**Key features:**
- Trajectory tracking: Controls the middle node to follow a circular arc path defined by `targetNodePosition.py`
- Iterative control: At each time step, iteratively adjusts the last two nodes' positions until the middle node reaches the target position within tolerance
- Applied force: Only gravity (beam weight) is applied to all nodes
- Boundary conditions: First node fixed in x and y; last two nodes are adjustable but constrained to maintain collinearity
- Number of nodes: Must be odd (to ensure a unique middle node)
- Plots at time steps: 200s, 400s, 600s, 800s, 1000s
- Outputs: Final node positions, angles, and coordinate trajectories over time

### Solver Modules

#### `solverNewtonRaphson.py`
Implements the implicit Newton-Raphson solver for time integration. This solver:
- Uses an implicit time integration scheme with Newton-Raphson iteration
- Solves the nonlinear system of equations iteratively until convergence
- Updates positions and velocities at each time step
- Handles boundary conditions through free DOF selection
- Returns updated positions and velocities for the next time step

**Function:** `solverNewtonRaphson(t_new, x_old, u_old, free_DOF, W, stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch, bending_stiffness_matrix, bending_index_matrix, l_k_bending, m, dt)`

#### `targetNodePosition.py`
Defines the target trajectory for the middle node. This module:
- Calculates the desired position of the middle node at a given time
- Uses a circular arc trajectory parameterized by time
- Returns target x and y coordinates based on the beam length and time

**Function:** `targetNodePosition(t_new, length)` - Returns `(target_x, target_y)`

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
- Applies gravitational force (weight) to all nodes in the system
- Calculates force based on node masses and gravitational acceleration (9.81 m/s²)
- Returns a force vector with forces in the negative y-direction (downward)

**Function:** `getExternalForce(m)` - Returns force vector `W` where `W[2*i+1] = -m[i] * 9.81` for each node `i`

### Initialization Module

#### `initSpringNetwork.py`
Initializes and creates the spring network data files. This script:
- Generates node coordinates along a beam of specified length
- Creates stretching springs connecting adjacent nodes
- Creates bending springs connecting every three consecutive nodes
- Distributes mass evenly across nodes
- Saves data to text files in the `springNetworkData/` directory

**Parameters (as defined in the script):**
- `N`: Number of nodes (default: 19, must be odd)
- `L`: Beam length in meters (default: 1.0 m)
- `E`: Elastic modulus (70e9 Pa for aluminum)
- `rho`: Density (2700 kg/m³ for aluminum)
- `r_outer`: Outer radius (0.013 m)
- `r_inner`: Inner radius (0.011 m)
- `EA`: Stretching stiffness (calculated from E and cross-sectional area)
- `EI`: Bending stiffness (calculated from E and moment of inertia)
- `m`: Total beam mass (calculated from density and volume)

**Note:** The script writes output files to `HW3/springNetworkData/` directory.

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

- **`plot_t_0.00s.png`**: Initial beam configuration at t=0s
- **`plot_t_200.00s.png`**: Beam configuration snapshot at t=200s
- **`plot_t_400.00s.png`**: Beam configuration snapshot at t=400s
- **`plot_t_600.00s.png`**: Beam configuration snapshot at t=600s
- **`plot_t_800.00s.png`**: Beam configuration snapshot at t=800s
- **`plot_t_1000.00s.png`**: Beam configuration snapshot at t=1000s
- **`final_node_positions.png`**: Trajectory of the last node's position (x-y plot)
- **`final_node_x_coordinate.png`**: X-coordinate of the last node over time
- **`final_node_y_coordinate.png`**: Y-coordinate of the last node over time
- **`final_node_angles.png`**: Angle of the last node over time (in radians)

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

- **Time step**: `dt = 1` second
- **Total simulation time**: `maxTime = 1000` seconds
- **Applied forces**: Gravity only (beam weight distributed across all nodes)
- **Target trajectory**: Circular arc path for middle node (defined by `targetNodePosition.py`)
- **Control tolerance**: `allowable_error = dist_btw_targets * 1e-2` (dynamically adjusted based on target spacing)
- **Maximum iterations**: 100 iterations per time step to reach target position
- **Boundary conditions**: 
  - First node: fixed in x and y directions
  - Last two nodes: Position adjustable but constrained to maintain collinearity with fixed end
  - All other nodes: Free to move

## Notes

- The simulation uses an implicit Newton-Raphson solver for stable time integration
- Rest lengths for stretching springs and rest curvatures for bending springs are calculated from the initial configuration
- The control algorithm iteratively adjusts the last two nodes' positions based on the error between the middle node's current position and target position
- The target trajectory follows a circular arc: `target_x = length/2 * cos(π/2 * t/1000)`, `target_y = -length/2 * sin(π/2 * t/1000)`
- The number of nodes must be odd to ensure a unique middle node exists

