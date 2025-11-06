import numpy as np

def getExternalForce(m):
    W = np.zeros_like(m) # number of DOFs in the system

    for i in range(m.shape[0] // 2):
        W[2 * i] = 0.0 # x-coordinate
        W[2 * i + 1] = -m[i] * 9.81 # y-coordinate

    return W