import numpy as np

def getExternalForce(m):
    W = np.zeros_like(m) # number of DOFs in the system

    for i in range(m.shape[0] // 3):
        W[3 * i] = 0.0 # x-coordinate
        W[3 * i + 1] = 0.0 # y-coordinate
        W[3 * i + 2] = -m[i] * 9.81 # z-coordinate

    return W