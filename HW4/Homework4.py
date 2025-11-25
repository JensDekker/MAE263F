import numpy as np
import os
import solverNewtonRaphson as snr
import plot as ploot
import matplotlib.pyplot as plt
import getExternalForce as gef
import mae263f_functions as mf
import time
import csv
import steady_state_checker as ssc
import format_number as fn

# store the start time
start_time = time.time()

nodes_filepath = 'HW4/springNetworkData/nodes.txt'
stretch_springs_filepath = 'HW4/springNetworkData/stretch_springs.txt'
bending_springs_filepath = "HW4/springNetworkData/bending_springs.txt"
twisting_springs_filepath = "HW4/springNetworkData/twisting_springs.txt"
parameters_filepath = 'HW4/springNetworkData/parameters.txt'


# Read the nodes file
node_coordinates = []
m_lst = []
with open(nodes_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 5:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            mass = float(parts[3])
            polar_mass = float(parts[4])

            node_coordinates.append([x, y, z])
            for i in range(3):
                m_lst.append(mass) # Added mass for each coordinate
            m_lst.append(polar_mass) # Added polar mass

        else:
            print("Skipped an invalid line.")

node_matrix = np.array(node_coordinates)
m_lst.pop() # Remove the last element (polar mass)
m = np.array(m_lst)
print(m)

print("Nodes successfully loaded.")

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

# Read the twisting springs file
index_info = []
stiffness_info = []
with open(twisting_springs_filepath, 'r') as f:
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

twisting_index_matrix = np.array(index_info)
twisting_stiffness_matrix = np.array(stiffness_info)

print("Twisting springs successfully loaded.")

# Read the parameters file
with open(parameters_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(':')]
        if parts[0] == 'f_char':
            f_char = float(parts[1])
        elif parts[0] == 'N':
            N = int(parts[1])
        elif parts[0] == 'nTurns':
            nTurns = int(parts[1])
        elif parts[0] == 'helix_dia':
            helix_dia = float(parts[1])
        elif parts[0] == 'p':
            p = float(parts[1])
        elif parts[0] == 'L_turn':
            L_turn = float(parts[1])
        elif parts[0] == 'L':
            L = float(parts[1])
        elif parts[0] == 'E':
            E = float(parts[1])
        elif parts[0] == 'rho':
            rho = float(parts[1])
        elif parts[0] == 'v_poisson':
            v_poisson = float(parts[1])
        elif parts[0] == 'A':
            A = float(parts[1])
        elif parts[0] == 'I':
            I = float(parts[1])
        elif parts[0] == 'J':
            J = float(parts[1])
        elif parts[0] == 'G':
            G = float(parts[1])
        elif parts[0] == 'EA':
            EA = float(parts[1])
        elif parts[0] == 'EI':
            EI = float(parts[1])
        elif parts[0] == 'GJ':
            GJ = float(parts[1])
        elif parts[0] == 'L_axial':
            L_axial = float(parts[1])
        # Note: 'm' parameter is skipped to avoid overwriting the mass array

print("Parameters successfully loaded.")

# output data file path with time the script started running
output_data_folder = 'HW4/plots'
if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)
# Include the helix diameter at the end of the filename
output_data_filepath = f'{output_data_folder}/output_data_{time.strftime("%Y%m%d_%H%M%S")}_{helix_dia:.3f}m.txt'

nv = N # number of vertices
ne = N - 1 # number of edges
ndof = 3 * nv + ne # number of degrees of freedom (3 translations per vertex + 1 rotation per edge)

# Initialize positions and velocities 
x_initial = np.zeros(ndof)
u_initial = np.zeros(ndof) # Assume initial velocity is 0

# Build x_initial from the nodes.txt file
for i in range(N):
    x_initial[4 * i] = node_matrix[i][0] # x-coordinate
    x_initial[4 * i + 1] = node_matrix[i][1] # y-coordinate
    x_initial[4 * i + 2] = node_matrix[i][2] # z-coordinate
    # assumes the initial rotation is 0

# mf.plotrod_simple(x_old, 0)

# Every spring has a rest length
l_k_stretch = np.zeros_like(stretch_stiffness_matrix)
for i in range(stretch_stiffness_matrix.shape[0]):
    ind = stretch_index_matrix[i].astype(int)
    xi = x_initial[ind[0]]
    yi = x_initial[ind[1]]
    xj = x_initial[ind[2]]
    yj = x_initial[ind[3]]
    l_k_stretch[i] = np.sqrt( (xj - xi) ** 2 + (yj - yi) ** 2 )

# Every bending spring has a rest curvature 
voronoiRefLen = np.zeros_like(bending_stiffness_matrix)
for i in range(bending_stiffness_matrix.shape[0]):
    ind = bending_index_matrix[i].astype(int)
    xi = x_initial[ind[0]]
    yi = x_initial[ind[1]]
    xj = x_initial[ind[2]]
    yj = x_initial[ind[3]]
    xk = x_initial[ind[4]]
    yk = x_initial[ind[5]]
    edge1 = np.sqrt( (xj - xi) ** 2 + (yj - yi) ** 2 )
    edge2 = np.sqrt( (xk - xj) ** 2 + (yk - yj) ** 2 )
    if i == 0 or i == bending_stiffness_matrix.shape[0] - 1:
        voronoiRefLen[i] = 0.5 * edge1
    else:
        voronoiRefLen[i] = 0.5 * (edge1 + edge2)

# Set up the Reference frames
tangent_initial = mf.computeTangent(x_initial)

t0 = tangent_initial[0, :]
arb_v = np.array([0, 0, -1])
a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3: # Check if t0 and arb_v are parallel
    arb_v = np.array([0, 1, 0])
    a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

a1_initial, a2_initial = mf.computeSpaceParallel(a1_first, x_initial)

# Material Frame
theta_initial = x_initial[3::4] # get all the theta angles
m1_initial, m2_initial = mf.computeMaterialDirectors(a1_initial, a2_initial, theta_initial)

# Natural Curvature and Twist
refTwist_initial = np.zeros(nv) 

# Natural Curvature
kappaBar_initial = mf.getKappa(x_initial, m1_initial, m2_initial)

# Natural Twist
twistBar_initial = np.zeros(nv)

# Time Scale and Time Parameters
time_scale = L / np.sqrt(E / rho)
max_dt = time_scale * 1e-1
min_dt = 4.4e-5
ndt_increments = 10
dt_inc = (max_dt - min_dt) / ndt_increments # time step size
# Validate the max_dt is greater than the min_dt
if max_dt < min_dt:
    print("Max time step is less than the min time step")
    exit()
# Simulations for Pt 1 and Pt 2 used:
# min_dt = 1e-7
# ndt_increments = 20

steady_time = 4 * time_scale
allowable_error = 5e-2 * L_axial
# Simulations for Pt 1 and Pt 2  used:
# steady_time = 5 * time_scale
# allowable_error = 1e-3 * L_axial

# Free DOFs
# Define the free DOFs by the list of fixed DOFs
# The first 7 DOFs are fixed (3 translations per vertex + 1 rotation per edge)
fixed_DOF = np.arange(7)
free_DOF = np.arange(ndof)
free_DOF = np.setdiff1d(free_DOF, fixed_DOF)

loads = np.array([1.0]) * f_char # loads to apply
# loads = np.logspace(-2, 1, 6) * f_char # loads to apply

W_mask = np.zeros(ndof)
W_mask[4*(nv-1) + 2] = 1 # Apply a force at the last node Z-direction
print(W_mask)

tol = EI / L**2 * 1e-3

# Damping coefficient for convergence analysis
mu_start = 1.81e-5 # Pa.s; Viscosity of air at 20°C
mu_increment = 1e-5 # Pa.s
mu_start = 0.0 # No damping


# Create the "working" variables
x_old = x_initial.copy()
u_old = u_initial.copy()
a1_old = a1_initial.copy()
a2_old = a2_initial.copy()
m1 = m1_initial.copy()
m2 = m2_initial.copy()  
kappaBar = kappaBar_initial.copy()
refTwist = refTwist_initial.copy()
twistBar = twistBar_initial.copy()

# Final height container
h_container = []

# write all positions to a file for plotting
with open(output_data_filepath, 'w') as f:
    fieldnames = ['Load', 'Time', "Time Step", 'Positions', 'Velocities', 'Flags']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over multiple loads
    row_data = {}
    for load in loads:
        W = load * W_mask
        row_data['Load'] = load
        mu = mu_start
        dt = min_dt
        ct = 0
        load_complete = False
        steady_state_flag = False

        flag = 1
        while not load_complete:
            row_data['Time Step'] = dt
            # Run simulation
            steady_state_reached = False
            while not steady_state_reached:
                print(f"Running simulation for load " + fn.format_number(load, decimals=3) 
                        + " at time " + fn.format_number(ct, decimals=4, sci_threshold_small=1e-2) 
                        + " with dt " + fn.format_number(dt, decimals=3))
                if ct == 0: # Write the initial data
                    row_data['Time'] = 0
                    row_data['Positions'] = x_old
                    row_data['Velocities'] = u_old
                    row_data['Flags'] = 1
                    writer.writerow(row_data)
                t_new = ct + dt
                row_data['Time'] = t_new


                # Call integrator
                x_new, u_new, a1_new, a2_new, flag = snr.solverNewtonRaphson(x_old, u_old, a1_old, a2_old, tol, W,
                                                       stretch_stiffness_matrix, l_k_stretch,  
                                                       bending_stiffness_matrix, voronoiRefLen, m1, m2, kappaBar,
                                                       twisting_stiffness_matrix, refTwist, twistBar,
                                                       m, dt, free_DOF, mu)
                if flag == -1:
                    print("Newton-Raphson did not converge")
                    mu += mu_increment
                    break
                row_data['Positions'] = x_new
                row_data['Velocities'] = u_new
                row_data['Flags'] = flag

                print(f"x_new: {x_new[-1]}")
                x_old = x_new
                u_old = u_new
                a1_old = a1_new
                a2_old = a2_new

                # Save the data
                writer.writerow(row_data)

                # Update the current time
                ct += dt
                
                # Check steady state with the NEW position (after solver and update)
                steady_state_reached = ssc.steady_state_checker(x_new, h_container, allowable_error, steady_time, dt)


            if flag == -1: # If the Newton-Raphson did not converge, increase the time step and move to the 
                dt += dt_inc

                # Reset the working variables
                x_old = x_initial.copy()
                u_old = u_initial.copy()
                a1_old = a1_initial.copy()
                a2_old = a2_initial.copy()
                m1 = m1_initial.copy()
                m2 = m2_initial.copy()  
                kappaBar = kappaBar_initial.copy()
                refTwist = refTwist_initial.copy()
                twistBar = twistBar_initial.copy()
                h_container = []
                ct = 0

                if dt > max_dt: # If the time step is greater than the max time step, break out of the loop
                    print("Maximum time step reached")
                    break
                continue
                
                    
            load_complete = True
        
        if flag == -1:
            print(f"Newton-Raphson did not converge through any time step for load {load}")
            break
        
        # reset the working variables
        x_old = x_initial.copy()
        u_old = u_initial.copy()
        a1_old = a1_initial.copy()
        a2_old = a2_initial.copy()
        m1 = m1_initial.copy()
        m2 = m2_initial.copy()  
        kappaBar = kappaBar_initial.copy()
        refTwist = refTwist_initial.copy()
        twistBar = twistBar_initial.copy()
        h_container = []
        ct = 0
        dt = min_dt
        
# store the end time
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")