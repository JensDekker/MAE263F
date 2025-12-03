import numpy as np
import os
import solverNewtonRaphson as snr
import plot as ploot
import matplotlib.pyplot as plt
import getExternalForce as gef
import mae263f_functions as mf
import time
import csv
import format_number as fn
import plotShell as ps
import plotShell_hinge as psh
import steady_state_checker as ssc

# store the start time
start_time = time.time()

nodes_filepath = 'HW5/springNetworkData/nodes.txt'
stretch_springs_filepath = 'HW5/springNetworkData/stretch_springs.txt'
bending_springs_filepath = "HW5/springNetworkData/bending_springs.txt"
twisting_springs_filepath = "HW5/springNetworkData/twisting_springs.txt"
parameters_filepath = 'HW5/springNetworkData/parameters.txt'


# Read the nodes file
node_coordinates = []
m_lst = []
with open(nodes_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 4:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            mass = float(parts[3])
            

            node_coordinates.append([x, y, z])
            for i in range(3):
                m_lst.append(mass) # Added mass for each coordinate

        else:
            print("Skipped an invalid line.")

node_matrix = np.array(node_coordinates)
m = np.array(m_lst)
mMat = np.diag(m)
print(m)

print("Nodes successfully loaded.")

# Read the stretching springs file
index_info = []
refLen_info = []
stiffness_info = []
with open(stretch_springs_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 4:
            first_node_idx = int(parts[0])
            second_node_idx = int(parts[1])
            refLen = float(parts[2])
            stiffness = float(parts[3])
            index_info.append([3 * first_node_idx, 
                               3 * first_node_idx + 1,
                               3 * first_node_idx + 2,
                               3 * second_node_idx, 
                               3 * second_node_idx + 1,
                               3 * second_node_idx + 2])
            refLen_info.append(refLen)
            stiffness_info.append(stiffness)

        else:
            print("Skipped an invalid line.")

stretch_index_matrix = np.array(index_info)
stretch_stiffness_matrix = np.array(stiffness_info)
refLen_matrix = np.array(refLen_info)

print("Springs successfully loaded.")

# Read the bending springs file
index_info = []
stiffness_info = []
with open(bending_springs_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 5:
            first_node_idx = int(parts[0])
            second_node_idx = int(parts[1])
            first_guide_node_idx = int(parts[2])
            second_guide_node_idx = int(parts[3])
            stiffness = float(parts[4])
            index_info.append([3 * first_node_idx, 
                               3 * first_node_idx + 1, 
                               3 * first_node_idx + 2, 
                               3 * second_node_idx, 
                               3 * second_node_idx + 1,
                               3 * second_node_idx + 2,
                               3 * first_guide_node_idx, 
                               3 * first_guide_node_idx + 1,
                               3 * first_guide_node_idx + 2,
                               3 * second_guide_node_idx, 
                               3 * second_guide_node_idx + 1,
                               3 * second_guide_node_idx + 2])
            stiffness_info.append(stiffness)
        else:
            print("Skipped an invalid line.")

bending_index_matrix = np.array(index_info)
bending_stiffness_matrix = np.array(stiffness_info)

print("Bending springs successfully loaded.")

# Read the parameters file
with open(parameters_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(':')]
        if parts[0] == 'nL_edge':
            nL_edge = int(parts[1])
        elif parts[0] == 'nW_edge':
            nW_edge = int(parts[1])
        elif parts[0] == 'length':
            length = float(parts[1])
        elif parts[0] == 'width':
            width = float(parts[1])
        elif parts[0] == 'thickness':
            thickness = float(parts[1])
        elif parts[0] == 'E':
            E = float(parts[1])
        elif parts[0] == 'I':
            I = float(parts[1])
        elif parts[0] == 'total_mass':
            total_mass = float(parts[1])
        elif parts[0] == 'N':
            N = int(parts[1])

print("Parameters successfully loaded.")

# output data file path with time the script started running
output_data_folder = 'HW5/plots'
if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)
# Include the helix diameter at the end of the filename
output_data_filepath = 'HW5/plots/output_data_0s-5s.txt'

nv = N # number of vertices
ne = N - 1 # number of edges
ndof = 3 * nv # number of degrees of freedom (3 translations per vertex + 1 rotation per edge)

# Initialize positions and velocities 
x_initial = np.zeros(ndof)
u_initial = np.zeros(ndof) # Assume initial velocity is 0

# Build x_initial from the nodes.txt file
for i in range(N):
    x_initial[3 * i] = node_matrix[i][0] # x-coordinate
    x_initial[3 * i + 1] = node_matrix[i][1] # y-coordinate
    x_initial[3 * i + 2] = node_matrix[i][2] # z-coordinate

ps.plotShell(x_initial, stretch_index_matrix, 0, output_dir=output_data_folder, show_plots=False)
psh.plotShell_hinge(x_initial, bending_index_matrix, 0, show_plots=True)

dt = 0.001
ct = 0
est_spring_constant = 3 * E * I / length**3
est_natural_freq = np.sqrt(est_spring_constant / total_mass)
time_scale = 1 / est_natural_freq

steady_time = 3 * time_scale
steady_tol = thickness * 1e-3
end_time = 120

sim_tol = bending_stiffness_matrix.max() / thickness * 1e-3

W = gef.getExternalForce(m)
visc = 0

fixedIndex = np.arange(12) # the first 4 nodes (12 dof) are fixed
freeIndex = np.arange(12, ndof)

steady_state_reached = False

x_old = x_initial.copy()
u_old = u_initial.copy()

print(stretch_index_matrix)
print(refLen_matrix)
print(bending_index_matrix)

nSteps = 0

h_container = []


# write all positions to a file for plotting
with open(output_data_filepath, 'w') as f:
    fieldnames = ['Time', 'Positions', 'Velocities']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over multiple loads
    row_data = {}

    while not steady_state_reached and ct < end_time:
        
        if ct == 0: # Write the initial data
            row_data['Time'] = 0
            row_data['Positions'] = x_old
            row_data['Velocities'] = u_old
            writer.writerow(row_data)
        t_new = ct + dt
        row_data['Time'] = ct
        # Call integrator
        x_new, u_new = snr.solverNewtonRaphson(x_old, u_old, freeIndex, dt, sim_tol, m, mMat,
                    stretch_stiffness_matrix, refLen_matrix, stretch_index_matrix,
                    bending_stiffness_matrix, bending_index_matrix,
                    W, visc)
        
        row_data['Positions'] = x_new
        row_data['Velocities'] = u_new
        
        x_old = x_new.copy()
        u_old = u_new.copy()
        # Save the data
        writer.writerow(row_data)

        # Update the current time
        ct += dt
        nSteps += 1
        if nSteps % 20 == 0:
            ps.plotShell(x_new, stretch_index_matrix, ct, output_dir=output_data_folder, show_plots=False)

        
        # Check steady state with the NEW position (after solver and update)
        steady_state_reached = ssc.steady_state_checker(x_new, [-4,-1], h_container, steady_tol, steady_time, dt)

        print(f"Time: {ct}")

sim_end_time = time.time()
print(f"Simulation completed in {sim_end_time - start_time} seconds.")
