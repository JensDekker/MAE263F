import numpy as np

def getExternalForce(node_matrix, magnitude):
    W = np.zeros(node_matrix.shape[0] * 2) # number of DOFs in the system

    # find the node with the x-coordinate closest to 0.75 meters
    closest_node_idx = np.argmin(np.abs(node_matrix[:, 0] - 0.75))

    # apply the force to the node
    W[2 * closest_node_idx] = 0.0 # x-coordinate
    W[2 * closest_node_idx + 1] = -magnitude # y-coordinate
    return W, closest_node_idx