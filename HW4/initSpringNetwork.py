import numpy as np
import os

def initSpringNetwork(N, nTurns, wire_dia, helix_dia, p,
                      EA, EI, GJ, m, output_dir):
    # N = number of nodes
    # L = length of the beam in meters
    # EA = stretching stiffness of the beam in N/m
    # Outputs:
    # Creates three files to store the nodes, stretching springs, and bending springs
    # nodes.txt: contains the coordinates of the nodes
    # stretching_springs.txt: contains the indices of the nodes connected by stretching springs
    # bending_springs.txt: contains the indices of the nodes connected by bending springs

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # Create nodes.txt file
    # Format: x_coord, y_coord where each line represents a node
    mass_per_node = m / (N - 1)

    start_angle = 0
    angle_step = 2 * np.pi * nTurns / (N - 1)
    end_angle = nTurns * 2 * np.pi + angle_step # add one more angle step to ensure the last node is at the end of the helix
    with open(os.path.join(output_dir, 'nodes.txt'), 'w') as f:
        for idx, arc_rad in enumerate(np.arange(start_angle, end_angle, angle_step)):
            x_coord = helix_dia / 2 * np.cos(arc_rad)
            y_coord = helix_dia / 2 * np.sin(arc_rad)
            z_coord = arc_rad / (2 * np.pi) * p
            f.write(f"{x_coord}, {y_coord}, {z_coord}")

            # Write the mass for the node
            if idx == 0:
                f.write(f", {mass_per_node/2}") # first node has half the mass of the other nodes
            elif idx == N - 1:
                f.write(f", {mass_per_node/2}") # last node has half the mass of the other nodes
            else:
                f.write(f", {mass_per_node}")
            
            # Write the polar mass for the node
            polar_mass = 0.5 * mass_per_node * wire_dia**2 / 4
            f.write(f", {polar_mass}\n")
            

    # Create stretching_springs.txt file
    # Format: node1_idx, node2_idx, stiffness where each line represents a stretching spring
    with open(os.path.join(output_dir, 'stretch_springs.txt'), 'w') as f:
        for i in range(N - 1):
            node1_idx = i
            node2_idx = i + 1
            stiffness = EA
            f.write(f"{node1_idx}, {node2_idx}, {stiffness}\n")
    
    # Create bending_springs.txt file
    # Format: node1_idx, node2_idx, node3_idx, stiffness where each line represents a bending spring
    # node2_idx is the node at the middle of the spring
    with open(os.path.join(output_dir, 'bending_springs.txt'), 'w') as f:
        for i in range(N - 2):
            node1_idx = i
            node2_idx = i + 1
            node3_idx = i + 2
            stiffness = EI
            f.write(f"{node1_idx}, {node2_idx}, {node3_idx}, {stiffness}\n")

    # Create twisting_springs.txt file
    # Format: node1_idx, node2_idx, node3_idx, stiffness where each line represents a twisting spring
    # node2_idx is the node at the middle of the spring
    with open(os.path.join(output_dir, 'twisting_springs.txt'), 'w') as f:
        for i in range(N - 2):
            node1_idx = i
            node2_idx = i + 1
            node3_idx = i + 2
            stiffness = GJ
            f.write(f"{node1_idx}, {node2_idx}, {node3_idx}, {stiffness}\n")

# Geometric properties 
N = 50
nTurns = 5
wire_dia = 0.002 # meters
helix_dia = 0.05 # meters
# helix diameters to test: 0.01, 0.013, 0.017, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05
# helix diameters tested: 0.04
p = wire_dia # drop's by the wire diameter for each turn
L_turn = np.sqrt((np.pi * helix_dia) ** 2 + p**2) # length of one turn of the helix
L = L_turn * nTurns # length of the helix

# Material properties
E = 10e6 # Pascals (e6 for easy conversion from MPa to Pa)
print(f'Debug - E: {E}')
rho = 7850 # kg/m^3
v_poisson = 0.5 # Poisson's ratio

# Cross-sectional area
A = np.pi * wire_dia**2 / 4 # meters^2

# Area moment of inertia
I = np.pi * wire_dia**4 / 64 # meters^4
print(f'Debug - I: {I}')

# Polar moment of inertia
J = np.pi * wire_dia**4 / 32 # meters^4
print(f'Debug - J: {J}')

# Shear Modulus
G = E / 2 / (1 + v_poisson) # meters^4
print(f'Debug - G: {G}')

# Stretching stiffness
EA = E * A
print(f'Debug - EA:{EA}') 

# Bending stiffness
EI = E * I
print(f'Debug - EI: {EI}')

# Twisting stiffness
GJ = E * J
print(f'Debug - GJ: {GJ}')

# Total Mass
m = rho * L * A

# Characteristic Force
f_char = E * I / L**2

# Axial Length
L_axial = L_turn * nTurns

output_dir = 'HW4/springNetworkData'

initSpringNetwork(N, nTurns, wire_dia, helix_dia, p,
                  EA, EI, GJ, m, output_dir)

with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
    f.write(f"N: {N}\n")
    f.write(f"nTurns: {nTurns}\n")
    f.write(f"wire_dia: {wire_dia}\n")
    f.write(f"helix_dia: {helix_dia}\n")
    f.write(f"p: {p}\n")
    f.write(f"L_turn: {L_turn}\n")
    f.write(f"L: {L}\n")
    f.write(f"E: {E}\n")
    f.write(f"rho: {rho}\n")
    f.write(f"v_poisson: {v_poisson}\n")
    f.write(f"A: {A}\n")
    f.write(f"I: {I}\n")
    f.write(f"J: {J}\n")
    f.write(f"G: {G}\n")
    f.write(f"EA: {EA}\n")
    f.write(f"EI: {EI}\n")
    f.write(f"GJ: {GJ}\n")
    f.write(f"m: {m}\n")
    f.write(f"f_char: {f_char}\n")
    f.write(f"L_axial: {L_axial}\n")

print(f'Parameters saved to: {os.path.join(output_dir, "parameters.txt")}')

