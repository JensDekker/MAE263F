import numpy as np

def computeMaterialDirectors(a1, a2, theta):
  # a1 = matrix of size ne x 3 - First reference director
  # a2 = matrix of size ne x 3
  # theta = vector of size ne (extracted from the DOF vector; every fourth element of the DOF vector)
  ne = len(theta) # Number of edges
  m1 = np.zeros_like(a1) # First material director
  m2 = np.zeros_like(a2) # Second material director
  for c in range(ne): # Loop over every edge
    cs = np.cos(theta[c])
    sn = np.sin(theta[c])
    m1[c, :] = cs * a1[c, :] + sn * a2[c, :]
    m2[c, :] = - sn * a1[c,:] + cs * a2[c, :]
  return m1, m2