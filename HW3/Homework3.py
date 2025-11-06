import numpy as np
import os
import solverNewtonRaphson as snr
import plot as ploot
import matplotlib.pyplot as plt
import getExternalForce as gef
import targetNodePosition as tnp

nodes_filepath = 'HW3/springNetworkData/nodes.txt'
stretch_springs_filepath = 'HW3/springNetworkData/stretch_springs.txt'
bending_springs_filepath = "HW3/springNetworkData/bending_springs.txt"
plot_filepath = 'HW3/plots'

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
# beam length is the distance between the first and last node
beam_length = np.sqrt((node_matrix[0][0] - node_matrix[-1][0]) ** 2 + (node_matrix[0][1] - node_matrix[-1][1]) ** 2)

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

N = node_matrix.shape[0] # number of nodes
ndof = 2 * N # number of degrees of freedom

# Error check that the number of nodes is odd
if N % 2 == 0:
    raise ValueError("The number of nodes must be odd")

# Initialize positions and velocities 
x_old = np.zeros(ndof)
u_old = np.zeros(ndof) # Assume initial velocity is 0

# Build x_old from the nodes.txt file
for i in range(N):
    x_old[2 * i] = node_matrix[i][0] # x-coordinate
    x_old[2 * i + 1] = node_matrix[i][1] # y-coordinate

# Save the initial configuration
ploot.plot(x_old, stretch_index_matrix, 0, save_plots=True, output_dir=plot_filepath)
 
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
    l_k_bending[i] = (edge1 + edge2) / 2

dt = 1 # time step size
maxTime = 1000 # total time of simulation in seconds
t = np.arange(0, maxTime + dt, dt) # time array

# Node of Interest: Middle Node
noi_idx = (N - 1) // 2
noi_x = x_old[2 * noi_idx]
noi_y = x_old[2 * noi_idx + 1]

# Free DOFs
# Define the free DOFs by the list of fixed DOFs
# First DOF, and last 2 Nodes are fixed
fixed_DOF = np.array([0,1,ndof-4,ndof-3,ndof-2,ndof-1])
free_DOF = np.arange(ndof)
free_DOF = np.setdiff1d(free_DOF, fixed_DOF)



print("Starting to solve for multiple loads...\n")

# Create a container to store the max deflection for each time step at a specific applied force


'''
Process Planning:

I want to implement a error function which iteratively updates the last node position based on the error between the middle node and the target position

The position of the last node is easy enough to guide, base it on the direction between the middle node and target position
The angle of the last node is trickier to optimize


Then comes the issue that the path planning is done at speed, not just fixed positions
Creating a stop-motion film will not solve the problem directly unless I factor in the different initial conditions for each time step

I know all the positions and velocities of the middle node, but not the rest of the beam, which makes this much trickier
Position is easy enough to error check, just look at where the middle node needs to be and where it actually is
Velocity can be error checked by looking at the previous time step's position and the current position

The best way to start the simulation is to stabilize the solution since the beam will sag slightly to start
The the last node's position and angle can be placed to the correct starting position for the middle node
Then the bulk of the simulation can begin running to copy the target path of the middle node
I can eliminate a target speed for the middle node, not really relevant to the problem

So no speed, settle the beam, move to starting position, then simulate the target path

This still leaves the problem of how to optimize the angle of the last node
Simplest would be to let it rotate freely, and record the angle at each time step

For the last position in the arc, I expect the beam to stretch out, meaning that the middle node will sit lower than the target position
In that case the last two nodes should point toward the pivot join to keep the beam in a compressed state
OR the beam could be bent more to curve the middle node to the target position

Best idea, test the localizing algorithm for the first target position to test was could be useful for the last node angle
---------------------------------

For each time step, need to iterate on the best last node position and angle to meet the target position of the middle node

'''

print('Processing force: Beam Weight...\n')

# Get the external force for this load
W = gef.getExternalForce(m)

# Reset initial conditions for this load
x_old = np.zeros(ndof)
u_old = np.zeros(ndof)
for i in range(N):
    x_old[2 * i] = node_matrix[i][0]
    x_old[2 * i + 1] = node_matrix[i][1]


tx0, ty0 = tnp.targetNodePosition(t[0], beam_length)
tx1, ty1 = tnp.targetNodePosition(t[1], beam_length)
dist_btw_targets = np.sqrt((tx1 - tx0) ** 2 + (ty1 - ty0) ** 2)
allowable_error = dist_btw_targets * 1e-2 # meter 
max_iterations = 100
iterations = 0

internode_length = beam_length / (N - 1)

# Create container to store the final node positions and angle for each time step once the target position is reached
final_node_positions = np.zeros((len(t), 2)) # x and y coordinates
final_node_positions[0,0] = x_old[-2]
final_node_positions[0,1] = x_old[-1]

final_node_angles = np.zeros(len(t)) # angle in radians

# Run simulation
for k in range(len(t) - 1):
    iterations = 0
    
    t_new = t[k+1]
    
    target_x, target_y = tnp.targetNodePosition(t_new, beam_length)

    distance_to_target = np.sqrt((noi_x - target_x) ** 2 + (noi_y - target_y) ** 2)

    while distance_to_target > allowable_error and iterations < max_iterations:
        
        # Call integrator
        x_new, u_new = snr.solverNewtonRaphson(t_new, x_old, u_old, free_DOF, W,
                                               stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch,
                                               bending_stiffness_matrix, bending_index_matrix, l_k_bending,
                                               m, dt)
        
        noi_x = x_new[2 * noi_idx]
        noi_y = x_new[2 * noi_idx + 1]
        
        target_x_delta = target_x - noi_x
        target_y_delta = target_y - noi_y

        distance_to_target = np.sqrt((target_x_delta) ** 2 + (target_y_delta) ** 2)
        # Only update the last two nodes if the distance to the target is greater than the allowable error
        if distance_to_target > allowable_error:
            
            # Determine the new positions for the last two nodes to meet the target position of the middle node
            new_fixed_DOF = x_old[fixed_DOF] + np.array([0, 0, # The first node position will not change
                                                         0, 0, # Will not update the second last node in this step
                                                         0.5*target_x_delta, 0.5*target_y_delta]) # The Last node position will update proportionally to the difference between the middle node and the target position

            # Align the last two nodes to be collinear with the fixed end
            theta = np.arctan2(new_fixed_DOF[5], new_fixed_DOF[4])

            # Update the second last node position
            new_fixed_DOF[2] = new_fixed_DOF[4] - internode_length * np.cos(theta) 
            new_fixed_DOF[3] = new_fixed_DOF[5] - internode_length * np.sin(theta)

            # Update the fixed DOFs to the new positions
            x_old[fixed_DOF] = new_fixed_DOF


        print(f'Time Step: {t_new}, X, Y, Distance to target: {target_x_delta}, {target_y_delta}, {distance_to_target}')
        iterations += 1
    # Plot at specific time steps

    # Update x_old and u_old
    x_old = x_new
    u_old = u_new

    if t_new in [200, 400, 600, 800, 1000]:
        ploot.plot(x_new, stretch_index_matrix, t_new, save_plots=True, output_dir=plot_filepath)
    final_node_positions[k+1,0] = x_old[-2]
    final_node_positions[k+1,1] = x_old[-1]
    final_node_angles[k+1] = theta


# Plots the final node positions
plt.figure()
plt.plot(final_node_positions[:,0], final_node_positions[:,1], 'bo-')
plt.title('Final Node Positions')
plt.xlabel('X-Coordinate')
plt.ylabel('Y-Coordinate')
plt.axis('equal')
# Save the plot
plt.savefig('HW3/plots/final_node_positions.png', dpi=300, bbox_inches='tight')
print(f'Plot saved to: HW3/plots/final_node_positions.png')

# PLot the x coordinate of the last node over time
plt.figure()
plt.plot(t, final_node_positions[:,0], 'bo-')
plt.title('X-Coordinate of the Last Node over Time')
plt.xlabel('Time [Second]')
plt.ylabel('X-Coordinate')
# Save the plot
plt.savefig('HW3/plots/final_node_x_coordinate.png', dpi=300, bbox_inches='tight')
print(f'Plot saved to: HW3/plots/final_node_x_coordinate.png')

# PLot the y coordinate of the last node over time
plt.figure()
plt.plot(t, final_node_positions[:,1], 'bo-')
plt.title('Y-Coordinate of the Last Node over Time')
plt.xlabel('Time [Second]')
plt.ylabel('Y-Coordinate')
# Save the plot
plt.savefig('HW3/plots/final_node_y_coordinate.png', dpi=300, bbox_inches='tight')
print(f'Plot saved to: HW3/plots/final_node_y_coordinate.png')

# Plots the final node angles
plt.figure()
plt.plot(final_node_angles, 'bo-')
plt.title('Final Node Angles')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
# Save the plot
plt.savefig('HW3/plots/final_node_angles.png', dpi=300, bbox_inches='tight')
print(f'Plot saved to: HW3/plots/final_node_angles.png')
