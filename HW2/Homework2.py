from typing import Any
import numpy as np
import os
import solverNewtonRaphson as snr
import solverExplicit as se
import plot as ploot
import matplotlib.pyplot as plt
import getExternalForce as gef

nodes_filepath = 'HW2/springNetworkData/nodes.txt'
stretch_springs_filepath = 'HW2/springNetworkData/stretch_springs.txt'
bending_springs_filepath = "HW2/springNetworkData/bending_springs.txt"


# Read the nodes file
node_coordinates = []
m_lst = []
with open(nodes_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 3:
            x = float(parts[0])
            mass = float(parts[2])
            m_lst.append(mass) # mass for the x-coord
            y = float(parts[1])
            m_lst.append(mass) # mass for the y-coord
            node_coordinates.append([x, y])

        else:
            print("Skipped an invalid line.")

node_matrix = np.array(node_coordinates)
m = np.array(m_lst)

print("Nodes successfully loaded.")
print(node_matrix)


# Read the stretching springs file
index_info = []
stiffness_info = []
with open(stretch_springs_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 3:
            first_node_idx = int(parts[0])
            second_node_idx = int(parts[1])
            stiffness = float(parts[2])
            index_info.append([2 * first_node_idx, 
                               2 * first_node_idx + 1, 
                               2 * second_node_idx, 
                               2 * second_node_idx + 1])
            stiffness_info.append(stiffness)

        else:
            print("Skipped an invalid line.")

stretch_index_matrix = np.array(index_info)
stretch_stiffness_matrix = np.array(stiffness_info)

print("Springs successfully loaded.")
print(stretch_index_matrix)
print(stretch_stiffness_matrix)

# Read the bending springs file
index_info = []
stiffness_info = []
with open(bending_springs_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 4:
            first_node_idx = int(parts[0])
            second_node_idx = int(parts[1])
            third_node_idx = int(parts[2])
            stiffness = float(parts[3])
            index_info.append([2 * first_node_idx, 
                               2 * first_node_idx + 1, 
                               2 * second_node_idx, 
                               2 * second_node_idx + 1,
                               2 * third_node_idx, 
                               2 * third_node_idx + 1])
            stiffness_info.append(stiffness)
        else:
            print("Skipped an invalid line.")

bending_index_matrix = np.array(index_info)
bending_stiffness_matrix = np.array(stiffness_info)

print("Bending springs successfully loaded.")
print(bending_index_matrix)
print(bending_stiffness_matrix)


N = node_matrix.shape[0] # number of nodes
ndof = 2 * N # number of degrees of freedom

# Initialize positions and velocities 
x_old = np.zeros(ndof)
u_old = np.zeros(ndof) # Assume initial velocity is 0

# Build x_old from the nodes.txt file
for i in range(N):
    x_old[2 * i] = node_matrix[i][0] # x-coordinate
    x_old[2 * i + 1] = node_matrix[i][1] # y-coordinate

# Save the initial configuration
ploot.plot(x_old, stretch_index_matrix, 0, save_plots=True, output_dir='HW1/plots')
 
# Every spring has a rest length
l_k_stretch = np.zeros_like(stretch_stiffness_matrix)
for i in range(stretch_stiffness_matrix.shape[0]):
    ind = stretch_index_matrix[i].astype(int)
    xi = x_old[ind[0]]
    yi = x_old[ind[1]]
    xj = x_old[ind[2]]
    yj = x_old[ind[3]]
    l_k_stretch[i] = np.sqrt( (xj - xi) ** 2 + (yj - yi) ** 2 )

# Every bending spring has a rest curvature 
l_k_bending = np.zeros_like(bending_stiffness_matrix)
for i in range(bending_stiffness_matrix.shape[0]):
    ind = bending_index_matrix[i].astype(int)
    xi = x_old[ind[0]]
    yi = x_old[ind[1]]
    xj = x_old[ind[2]]
    yj = x_old[ind[3]]
    xk = x_old[ind[4]]
    yk = x_old[ind[5]]
    edge1 = np.sqrt( (xj - xi) ** 2 + (yj - yi) ** 2 )
    edge2 = np.sqrt( (xk - xj) ** 2 + (yk - yj) ** 2 )
    l_k_bending[i] = edge1 + edge2 / 2


dt = 1e-2 # time step size
maxTime = 1 # total time of simulation in seconds
t = np.arange(0, maxTime + dt, dt) # time array


# Free DOFs
# Define the free DOFs by the list of fixed DOFs
fixed_DOF = np.array([0,1,ndof-1]) # Restricts the 1st node's x and y coordinates, and the last node's y coordinate
free_DOF = np.arange(ndof)
free_DOF = np.setdiff1d(free_DOF, fixed_DOF)

# Multiple applied forces
# applied_forces = np.array([10, 20, 50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 20000, 50000, 70000, 100000, 200000, 500000, 700000, 1000000])  # N
applied_forces = np.arange(500, 20500, 500)  # N
expected_deflections = np.zeros_like(applied_forces, dtype=float)
simulated_deflections = np.zeros_like(applied_forces, dtype=float)

# Expected deflection calculation setup
beam_length = 1 - l_k_stretch[0]  # meter

# Find where the force will be applied (only need to do once)
closest_node_idx = np.argmin(np.abs(node_matrix[:, 0] - 0.75))
applied_force_location = beam_length - node_matrix[closest_node_idx][0] 
print(f'The applied force location is: {applied_force_location}')

print("Starting to solve for multiple loads...\n")

# Loop over different applied forces
for force_idx, applied_force in enumerate(applied_forces):
    print(f'Processing force: {applied_force} N')
    
    # Get the external force for this load
    W, _ = gef.getExternalForce(node_matrix, applied_force)
    
    # Calculate expected deflection for this load
    expected_deflection = - applied_force * applied_force_location / ( 9 * np.sqrt(3) * beam_length * bending_stiffness_matrix[0] ) * (beam_length ** 2 - applied_force_location ** 2) ** 1.5
    expected_deflections[force_idx] = expected_deflection
    
    # Reset initial conditions for this load
    x_old = np.zeros(ndof)
    u_old = np.zeros(ndof)
    for i in range(N):
        x_old[2 * i] = node_matrix[i][0]
        x_old[2 * i + 1] = node_matrix[i][1]
    
    # Run simulation
    for k in range(len(t) - 1):
        t_new = t[k+1]
        
        # Call integrator
        x_new, u_new = snr.solverNewtonRaphson(t_new, x_old, u_old, free_DOF, W,
                                               stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch,
                                               bending_stiffness_matrix, bending_index_matrix, l_k_bending,
                                               m, dt)
        
        # Plot at specific time steps
        # To plot for ALL forces, change "force_idx == 0" to "True"
        # To plot only for the FIRST force, keep "force_idx == 0"
        if force_idx == 0 and t_new in [0.25, 0.5, 0.75, 1.0]:
            expected_max_deflection_location = np.sqrt((beam_length ** 2 - applied_force_location ** 2) / 3)
            ploot.plot(x_new, stretch_index_matrix, t_new, save_plots=True, output_dir='HW2/plots', 
                       expected_deflection=expected_deflection, 
                       expected_max_deflection_location=expected_max_deflection_location,
                       applied_force=applied_force)
        
        # Update x_old and u_old
        x_old = x_new
        u_old = u_new
    
    # Store the final simulated deflection (minimum y-coordinate)
    print(f'  Debug - x_old[free_DOF[1::2]]: {x_old[free_DOF[1::2]]}')
    print(f'  Debug - np.min(x_old[free_DOF[1::2]]): {np.min(x_old[free_DOF[1::2]])}')
    simulated_deflections[force_idx] = np.min(x_old[free_DOF[1::2]])
    print(f'  Debug - simulated_deflections: {simulated_deflections}')
    print(f'  Expected deflection: {expected_deflection:.6f} m')
    print(f'  Simulated deflection: {simulated_deflections[force_idx]:.6f} m\n')

# Create comparison plot
plt.figure(figsize=(10, 6))
plt.plot(applied_forces, expected_deflections, 'ro-', label='Expected Deflection', linewidth=2, markersize=8)
plt.plot(applied_forces, simulated_deflections, 'bs-', label='Simulated Deflection', linewidth=2, markersize=8)
plt.title('Expected vs Simulated Deflection for Multiple Applied Loads')
# plt.xscale('log')
plt.xlabel('Applied Force [N]')
plt.ylabel('Deflection [m]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(min(applied_forces) - 200, max(applied_forces) + 200)

if not os.path.exists('HW2/plots'):
    os.makedirs('HW2/plots')

filename = "HW2/plots/plot_expected_vs_simulated.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f'Comparison plot saved to: {filename}')

# Print results table
print("\nResults Summary:")
print("Force [N]\tExpected [m]\tSimulated [m]\tError [%]")
print("-" * 60)
# The percentage error is calculated as the absolute difference between the expected and simulated deflections divided by the absolute value of the expected deflection, multiplied by 100.
percentage_error = abs(expected_deflections - simulated_deflections) / abs(expected_deflections) * 100


for i in range(len(applied_forces)):
    print(f"{applied_forces[i]:.0f}\t\t{expected_deflections[i]:.6f}\t{simulated_deflections[i]:.6f}\t{percentage_error[i]:.2f}")

# Plot percentage error vs applied force
plt.figure(figsize=(10, 6))
plt.plot(applied_forces, percentage_error, 'ro-', label='Percentage Error', linewidth=2, markersize=8)
plt.title('Percentage Error vs Applied Force')
plt.xscale('log')
plt.xlabel('Applied Force [N]')
plt.ylabel('Percentage Error [%]')
plt.legend()
plt.xlim(min(applied_forces) - 200, max(applied_forces) + 200)
plt.savefig('HW2/plots/plot_percentage_error.png', dpi=300, bbox_inches='tight')
print(f'Percentage error plot saved to: {filename}')