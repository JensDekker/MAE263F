import numpy as np
from .computeReferenceTwist import computeReferenceTwist

def getRefTwist(a1, tangent, refTwist = None):

    # Given all the reference frames along the rod, we calculate the reference
    # twist along the rod on every node.

    ne = a1.shape[0] # ne is number of edges. Shape of a1 is ne x 3
    nv = ne + 1  # nv is number of nodes

    if refTwist is None: # No guess is provided
      refTwist = np.zeros(nv) # Intialize to all zeros.

    for c in np.arange(1,ne): # All internal nodes (i.e., all nodes except terminal nodes)
        a1e = a1[c-1,0:3]
        a1f = a1[c,  0:3]
        t1 =  tangent[c-1,0:3]
        t2 =  tangent[c,  0:3]
        refTwist[c] = computeReferenceTwist(a1e, a1f, t1, t2, refTwist[c])
    return refTwist