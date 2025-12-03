import numpy as np
from .computeKappa import computeKappa

def getKappa(q, m1, m2):
    nv = (len(q) + 1) // 4  # nv is number of nodes
    ne = nv - 1  # ne is number of edges

    kappa = np.zeros((nv, 2))  # Initialize kappa array

    for c in range(2, nv):  # Loop over edges (from second to last)

        # Extract node positions from q
        node0 = q[4*c-8:4*c-5]
        node1 = q[4*c-4:4*c-1]
        node2 = q[4*c+0:4*c+3]

        # Extract m1 and m2 for the current and previous edges
        m1e = m1[c-2,:].flatten()  # m1 vector on c-1 th edge
        # Another option is m1e = np.squeeze(np.array(m1[c-2, :]))
        m2e = m2[c-2,:].flatten()  # m2 vector on c-1 th edge
        m1f = m1[c-1,:].flatten()  # m1 vector on c th edge
        m2f = m2[c-1,:].flatten()  # m2 vector on c th edge

        # Compute local curvature at each node
        kappa_local = computeKappa(node0, node1, node2, m1e, m2e, m1f, m2f)

        # Store the curvature values
        kappa[c-1, 0] = kappa_local[0]
        kappa[c-1, 1] = kappa_local[1]

    return kappa