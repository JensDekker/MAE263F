import numpy as np

def initSpringNetwork(N, L, EA, EI, m):
    # N = number of nodes
    # L = length of the beam in meters
    # EA = stretching stiffness of the beam in N/m
    # Outputs:
    # Creates three files to store the nodes, stretching springs, and bending springs
    # nodes.txt: contains the coordinates of the nodes
    # stretching_springs.txt: contains the indices of the nodes connected by stretching springs
    # bending_springs.txt: contains the indices of the nodes connected by bending springs

    # Create nodes.txt file
    # Format: x_coord, y_coord where each line represents a node
    mass_per_node = m / (N - 1)
    with open('HW2/springNetworkData/nodes.txt', 'w') as f:
        for i in range(N):
            x_coord = i / (N - 1) * L
            f.write(f"{x_coord}, 0, {mass_per_node}\n")

    # Create stretching_springs.txt file
    # Format: node1_idx, node2_idx, stiffness where each line represents a stretching spring
    with open('HW2/springNetworkData/stretch_springs.txt', 'w') as f:
        for i in range(N - 1):
            node1_idx = i
            node2_idx = i + 1
            stiffness = EA
            f.write(f"{node1_idx}, {node2_idx}, {stiffness}\n")
    
    # Create bending_springs.txt file
    # Format: node1_idx, node2_idx, node3_idx, stiffness where each line represents a bending spring
    # node2_idx is the node at the middle of the spring
    with open('HW2/springNetworkData/bending_springs.txt', 'w') as f:
        for i in range(N - 2):
            node1_idx = i
            node2_idx = i + 1
            node3_idx = i + 2
            stiffness = EI
            f.write(f"{node1_idx}, {node2_idx}, {node3_idx}, {stiffness}\n")


N = 50
L = 1.0 # meters
E = 70e9 # Aluminum in Pascals (e9 for easy conversion from GPa to Pa)
print(f'Debug - E: {E}')
rho = 2700 # Aluminum density in kg/m^3
 
r_outer = 0.013 # meters
r_inner = 0.011 # meters

# Cross-sectional area
A = np.pi * (r_outer**2 - r_inner**2) # meters^2

# Bending stiffness
I = np.pi * (r_outer**4 - r_inner**4) / 4 # meters^4
print(f'Debug - I: {I}')

# Stretching stiffness
EA = E * A

# Bending stiffness
EI = E * I
print(f'Debug - EI: {EI}')

# Total Beam Mass
m = rho * L * A

initSpringNetwork(N, L, EA, EI, m)