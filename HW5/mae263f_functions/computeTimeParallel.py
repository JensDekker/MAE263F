import numpy as np
from .computeTangent import computeTangent
from .parallel_transport import parallel_transport

def computeTimeParallel(a1_old, q0, q):
  # a1_old: First time parallel frame director in "old" configuration
  # q0: "old" shape of the rod or DOF vector
  # q: "new" shape (a1 on this new shape is unknown)

  nv = (len(q)+1) // 4
  ne = nv -1

  tangent0 = computeTangent(q0) # Get the tangents in old configuration
  tangent = computeTangent(q) # Get the tangents in new configuration

  a1 = np.zeros((ne, 3)) # First time parallel frame director
  a2 = np.zeros((ne, 3)) # Second time parallel frame director

  for c in np.arange(ne): # Loop over every edge
    t0 = tangent0[c,:] # old tangent on the c-th edge
    t1 = tangent[c,:] # new tangent on the c-th edge
    a1[c,:] = parallel_transport(a1_old[c,:], t0, t1)
    a1[c,:] = a1[c,:] - np.dot(a1[c,:], t1) * t1 # Ensure it is orthogonal to t1
    a1[c, :] = a1[c,:] / np.linalg.norm(a1[c,:]) # Ensure it is unit
    a2[c, :] = np.cross(t1, a1[c,:])
    a2[c,:] = a2[c,:] - np.dot(a2[c,:], t1) * t1 # Ensure it is orthogonal to t1
    a2[c, :] = a2[c,:] / np.linalg.norm(a2[c,:]) # Ensure it is unit

  return a1, a2