import numpy as np
import solverNewtonRaphson as snr
import plot as ploot
import matplotlib.pyplot as plt

nodes_filepath = 'HW1/nodes.txt'
springs_filepath = 'HW1/springs.txt'

# Read the nodes file
node_coordinates = []
with open(nodes_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(',')]

        if len(parts) == 2:
            x = int(parts[0])
            y = int(parts[1])
            node_coordinates.append([x, y])

        else:
            print("Skipped an invalid line.")

node_matrix = np.array(node_coordinates)

print("Nodes successfully loaded.")
print(node_matrix)


# Read the springs file
index_info = []
stiffness_info = []
with open(springs_filepath, 'r') as f:
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

index_matrix = np.array(index_info)
stiffness_matrix = np.array(stiffness_info)

print("Springs successfully loaded.")
print(index_matrix)
print(stiffness_matrix)

N = node_matrix.shape[0] # number of nodes
ndof = 2 * N # number of degrees of freedom

# Initialize positions and velocities 
x_old = np.zeros(ndof)
u_old = np.zeros(ndof)

# Build x_old from the nodes.txt file
for i in range(N):
    x_old[2 * i] = node_matrix[i][0] # x-coordinate
    x_old[2 * i + 1] = node_matrix[i][1] # y-coordinate

# Save the initial configuration
ploot.plot(x_old, index_matrix, 0, save_plots=True, output_dir='HW1/plots')
 
# Every spring has a rest length
l_k = np.zeros_like(stiffness_matrix)
for i in range(stiffness_matrix.shape[0]):
    ind = index_matrix[i].astype(int)
    xi = x_old[ind[0]]
    yi = x_old[ind[1]]
    xj = x_old[ind[2]]
    yj = x_old[ind[3]]
    l_k[i] = np.sqrt( (xj - xi) ** 2 + (yj - yi) ** 2 )

# Mass 
m = np.ones(ndof)
m *= 1 # Every point has 1 kg of mass

dt = 0.1 # time step size
maxTime = 100 # total time of simulation in seconds
t = np.arange(0, maxTime + dt, dt) # time array

# Free DOFs
# Set which nodes are free in the system
free_nodes = [1, 3]
free_DOF = np.zeros(len(free_nodes) * 2, dtype=int)
# stores the indices of the free DOFs in the system
for i in range(len(free_nodes)):
    free_DOF[2 * i] = 2 * free_nodes[i]
    free_DOF[2 * i + 1] = 2 * free_nodes[i] + 1

# Container to store the y-coordinate of the middle node
y_middle = np.zeros(len(t))
y_middle[0] = x_old[3] # y-coordinate of the second node

# Container to store the y-coordinate of the free nodes
y_free = np.zeros((len(free_nodes), len(t)))
for i in range(len(free_nodes)):
    y_free[i,0] = x_old[2 * free_nodes[i] + 1]

for k in range(len(t) - 1):
    t_new = t[k+1] # Time
    
    # Call integrator
    x_new, u_new = snr.solverNewtonRaphson(t_new, x_old, u_old, free_DOF, 
                                           stiffness_matrix, index_matrix,
                                           m, dt, l_k)
    
    if t_new in [0, 0.1, 1.0, 10.0, 100.0]:
        ploot.plot(x_new, index_matrix, t_new, save_plots=True, output_dir='HW1/plots')
    y_middle[k+1] = x_new[3]

    # Update the y-coordinate of the free nodes
    for i in range(len(free_nodes)):
        y_free[i,k+1] = x_new[2 * free_nodes[i] + 1]

    # Update x_old and u_old
    x_old = x_new
    u_old = u_new

# Plot y_middle
plt.figure()
plt.plot(t, y_middle, 'ro-')
plt.xlabel('Time [Second]')
plt.ylabel('Y-Coordinate of the Second Node [Meter]')
plt.show()

# Plot y_free, with different colors for each free node
plt.figure()
for i in range(len(free_nodes)):
    plt.plot(t, y_free[i,:], f'bo-')
plt.xlabel('Time [Second]')
plt.ylabel('Y-Coordinate of the Free Nodes [Meter]')
plt.show()

