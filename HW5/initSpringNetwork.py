import numpy as np
import os

def initSpringNetwork(nL_edge, nW_edge, beam_length, beam_width, thickness, m, output_dir):
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

    mesh_size = {
        'L': nL_edge + 2, # +1 for edges to nodes, and +1 for the fixed nodes
        'W': nW_edge + 1 # +1 for edges to nodes
    } # this is for the number of nodes in the x and y directions

    N = mesh_size['L'] * mesh_size['W']

    # Create nodes.txt file
    # Format: x_coord, y_coord where each line represents a node
    mass_per_node = m / N

    Lstep = beam_length / nL_edge
    Wstep = beam_width / nW_edge
    L_start = -Lstep # start of the beam one Lstep left for the fixed nodes
    W_start = beam_width # start of the beam one Wstep up for the fixed nodes

    # create the diagonal indexing mask
    diagonal_mask_delta = np.zeros(mesh_size['L'] + 1)
    for i in range(mesh_size['L'] + 1):
        if i % 2 == 0:
            diagonal_mask_delta[i] = 1
        else:
            diagonal_mask_delta[i] = -1
    diagonal_mask_delta = diagonal_mask_delta + mesh_size['W']
    
    # guide node mask for the bending hinges
    guide_node_mask = np.zeros(mesh_size['L'] + 1)
    for i in range(mesh_size['L'] + 1):
        if i % 2 == 0:
            guide_node_mask[i] = 1
        else:
            guide_node_mask[i] = -1
    


    # Moves from the upper left most node, down, and then right
    nodes_list = []
    N = 0
    with open(os.path.join(output_dir, 'nodes.txt'), 'w') as f:
        z_coord = 0.0
        for i in range(mesh_size['L']):
            x_coord = L_start + i * Lstep
            for j in range(mesh_size['W']):
                y_coord = W_start - j * Wstep
                f.write(f'{x_coord}, {y_coord}, {z_coord}, {mass_per_node}\n')
                nodes_list.append(np.array([x_coord, y_coord, z_coord]))
                N += 1
    
    with open(os.path.join(output_dir, 'stretch_springs.txt'), 'w') as f:
        # The index of the nodes starts from 0
        # all the edges will go around the perimeter of the mesh and the interconnects

        # All width-wise stretching springs
        for i in range(mesh_size['L']):
            for j in range(mesh_size['W'] - 1):
                first_node_idx = i * mesh_size['W'] + j
                second_node_idx = i * mesh_size['W'] + j + 1
                x_0 = nodes_list[first_node_idx]
                x_1 = nodes_list[second_node_idx]
                refLen = np.linalg.norm(x_1 - x_0)
                stretching_stiffness = np.sqrt(3) / 2 * E * thickness * refLen**2
                f.write(f'{first_node_idx}, {second_node_idx}, {refLen}, {stretching_stiffness}\n')
        
        # All length-wise stretching springs
        for i in range(mesh_size['W']):

            # diagonal stretching springs initialization
            if i != mesh_size['W'] - 1: # not the last
                starting_node = i
                if starting_node % 2 == 1:
                    starting_node += 1
                    working_mask = diagonal_mask_delta[1:]
                else:
                    working_mask = diagonal_mask_delta[:-1]
                diagonal_springs_idx = [starting_node]
                for k in range(1, len(working_mask)):
                    delta_sum = working_mask[0:k].sum()
                    diagonal_springs_idx.append(starting_node + delta_sum)
                diagonal_springs_idx = np.array(diagonal_springs_idx).astype(int)
                print(diagonal_springs_idx)
            
            for j in range(mesh_size['L'] - 1):
                # length-wise stretching springs
                first_node_idx = int(i + (j * mesh_size['W']))
                second_node_idx = int(i + ((j + 1) * mesh_size['W']))
                x_0 = nodes_list[first_node_idx]
                x_1 = nodes_list[second_node_idx]
                refLen = np.linalg.norm(x_1 - x_0)
                stretching_stiffness = np.sqrt(3) / 2 * E * thickness * refLen**2
                f.write(f'{first_node_idx}, {second_node_idx}, {refLen}, {stretching_stiffness}\n')

                # diagonal stretching springs
                if i != mesh_size['W'] - 1: # not the last
                    first_node_diag_idx = int(diagonal_springs_idx[j])
                    second_node_diag_idx = int(diagonal_springs_idx[j + 1])
                    x_d0 = nodes_list[first_node_diag_idx]
                    x_d1 = nodes_list[second_node_diag_idx]
                    refLen = np.linalg.norm(x_d1 - x_d0)
                    stretching_stiffness = np.sqrt(3) / 2 * E * thickness * refLen**2
                    f.write(f'{first_node_diag_idx}, {second_node_diag_idx}, {refLen}, {stretching_stiffness}\n')
    
    with open(os.path.join(output_dir, 'bending_springs.txt'), 'w') as f:
        # diagonal stretching springs initialization
        for i in range(mesh_size['W']):

            # diagonal stretching springs initialization
            if i != mesh_size['W'] - 1: # not the last
                starting_node = i
                if starting_node % 2 == 1:
                    starting_node += 1
                    working_mask = diagonal_mask_delta[1:]
                else:
                    working_mask = diagonal_mask_delta[:-1]
                bending_springs_idx = [starting_node]
                for k in range(1, len(working_mask)):
                    delta_sum = working_mask[0:k].sum()
                    bending_springs_idx.append(starting_node + delta_sum)
                bending_springs_idx = np.array(bending_springs_idx).astype(int)
                print(bending_springs_idx)
            
            for j in range(mesh_size['L'] - 1):
                if i != mesh_size['W'] - 1: # not the last
                    first_node_diag_idx = int(bending_springs_idx[j])
                    second_node_diag_idx = int(bending_springs_idx[j + 1])
                    first_guide_node_idx = int(guide_node_mask[j]) + first_node_diag_idx
                    second_guide_node_idx = int(guide_node_mask[j + 1]) + second_node_diag_idx

                    bending_stiffness = 2.0 / np.sqrt(3.0) * E * thickness**3.0 / 12.0
                    f.write(f'{first_node_diag_idx}, {second_node_diag_idx}, {first_guide_node_idx}, {second_guide_node_idx}, {bending_stiffness}\n')
        
        # width-wise bending springs
        for i in range(1,mesh_size['L']-1):
            for j in range(0, mesh_size['W']-1):
                node_loc_idx = i + j
                hinge_start = i * mesh_size['W'] + j # the index of the starting node of the hinge
                hinge_end = hinge_start + 1 # moving along the width
                if node_loc_idx % 2 == 1:
                    l_hinge_guide_node_idx = hinge_start - mesh_size['W'] # move one node to the left
                    r_hinge_guide_node_idx = hinge_start + mesh_size['W'] # move one node to the right
                else:
                    l_hinge_guide_node_idx = hinge_end - mesh_size['W'] # move one node to the left
                    r_hinge_guide_node_idx = hinge_end + mesh_size['W'] # move one node to the right
                bending_stiffness = 2.0 / np.sqrt(3.0) * E * thickness**3.0 / 12.0
                f.write(f'{hinge_start}, {hinge_end}, {l_hinge_guide_node_idx}, {r_hinge_guide_node_idx}, {bending_stiffness}\n') 

                
        
# Material properties
E = 10e6 # Pascals (e6 for easy conversion from MPa to Pa)
rho = 1000 # kg/m^3

# Geometric properties
thickness = 0.002 # meters
width = 0.01 # meters
length = 0.1 # meters
nL_edge = 8
nW_edge = 1

# Area
A = width * thickness

# Area moment of inertia
I = width * thickness**3 / 12

# Stretching stiffness
EA = E * A

# Bending stiffness
EI = E * I

# Distributed load
q = rho * A * -9.81 # N/m

# Total Mass
total_mass = rho * length * width * thickness

# number of nodes
N = (nL_edge + 2) * (nW_edge + 1)

output_dir = 'HW5/springNetworkData'

initSpringNetwork(nL_edge, nW_edge, length, width, thickness, total_mass, output_dir)

with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
    f.write(f"nL_edge: {nL_edge}\n")
    f.write(f"nW_edge: {nW_edge}\n")
    f.write(f"length: {length}\n")
    f.write(f"width: {width}\n")
    f.write(f"thickness: {thickness}\n")
    f.write(f"E: {E}\n")
    f.write(f"rho: {rho}\n")
    f.write(f"A: {A}\n")
    f.write(f"I: {I}\n")
    f.write(f"EA: {EA}\n")
    f.write(f"EI: {EI}\n")
    f.write(f"total_mass: {total_mass}\n")
    f.write(f"N: {N}\n")
    f.write(f"q: {q}\n")

print(f'Parameters saved to: {os.path.join(output_dir, "parameters.txt")}')