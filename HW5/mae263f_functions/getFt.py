import numpy as np
from .gradEt_hessEt import gradEt_hessEt

def getFt(q, refTwist, twistBar, GJ, voronoiRefLen):
  # TWISTING
  # Input q is a DOF vector of size 4*nv - 1
  ndof = len(q)
  nv = (ndof + 1) // 4
  ne = nv - 1

  Ft = np.zeros(ndof)
  Jt = np.zeros((ndof, ndof))

  # Loop over each twisting spring
  for c in range(1, ne): # Ignore the terminal nodes (0 and nv)
    node0 = q[4 * (c - 1) : 4 * (c - 1) + 3] # (c-1)-th node
    node1 = q[4 * c : 4*c + 3] # c-th node
    node2 = q[4 * (c + 1) : 4 * (c + 1) + 3] # (c+1)-th node
    theta_e = q[4*c-1]
    theta_f = q[4*c+3]
    # Calculate twisting force due to the turning at the c-th node

    dL = voronoiRefLen[c-1]

    dF, dJ = gradEt_hessEt(node0, node1, node2, theta_e, theta_f, refTwist[c], twistBar[c], dL, GJ[c-1])
    ind = np.array([4*c-4, 4*c-3, 4*c-2, 4*c-1,4*c, 4*c+1, 4*c+2, 4*c+3, 4*c+4, 4*c+5, 4*c+6])

    Ft[ind] -= dF
    Jt[np.ix_(ind, ind)] -= dJ

  return Ft, Jt