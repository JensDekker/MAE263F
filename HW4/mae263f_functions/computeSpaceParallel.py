import numpy as np
from .computeTangent import computeTangent
from .parallel_transport import parallel_transport

def computeSpaceParallel(u1_first, q):
  # u1_first = first reference frame vector (arbitrary but orthonormal adapted) on the first edge
  # q is the DOF vector of size 4*nv - 1
  nv = (len(q)+1) // 4
  ne = nv -1

  tangent = computeTangent(q) # Get the tangent of each edge

  u1 = np.zeros((ne, 3)) # First reference frame director
  u2 = np.zeros((ne, 3)) # Second reference frame director

  u1_first = u1_first / np.linalg.norm(u1_first) # Ensure it is unit

  # First edge
  u1[0,:] = u1_first
  t0 = tangent[0,:]
  u2[0,:] = np.cross(t0, u1_first)
  u2[0,:] = u2[0,:] / np.linalg.norm(u2[0,:]) # Ensure it is unit

  for c in np.arange(1, ne):
    t0 = tangent[c-1,:] # "From" tangent
    t1 = tangent[c,:] # "To" tangent
    u1[c,:] = parallel_transport(u1[c-1,:], t0, t1)
    u1[c, :] = u1[c,:] / np.linalg.norm(u1[c,:]) # Ensure it is unit
    u2[c,:] = np.cross(t1, u1[c,:])
    u2[c, :] = u2[c,:] / np.linalg.norm(u2[c,:]) # Ensure it is unit

  return u1, u2