import numpy as np

def computeTangent(q):
  # q is the DOF vector of size 4*nv - 1 = 3 * nv + ne
  nv = (len(q) + 1) // 4
  ne = nv - 1
  tangent = np.zeros((ne, 3))  # every edge has a tangent
  for c in range(ne):
    node0 = q[4*c:4*c+3]
    node1 = q[4*c+4:4*c+7]
    edge = node1 - node0
    tangent[c, :] = edge / np.linalg.norm(edge)
  return tangent