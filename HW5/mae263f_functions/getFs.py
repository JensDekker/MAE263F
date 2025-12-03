import numpy as np
from .gradEs_hessEs import gradEs_hessEs

def getFs(q, EA, refLen):
  # STRETCHING
  # Input q is a DOF vector of size 4*nv - 1
  # Output is the stretching force vector (size 4*nv-1) and it's gradient w.r.t. q (square matrix)
  ndof = len(q)
  nv = (ndof + 1) // 4
  ne = nv - 1

  Fs = np.zeros(ndof)
  Js = np.zeros((ndof, ndof))

  # Loop over each stretching spring
  for c in range(ne):
    xa = q[4 * c : 4 * c + 3]
    xb = q[4 * c + 4 : 4 * c + 7]
    ind = np.array([4*c, 4*c+1, 4*c+2, 4*c+4, 4*c+5, 4*c+6])
    dL = refLen[c]

    dF, dJ = gradEs_hessEs(xa, xb, dL, EA[c])

    Fs[ind] -= dF
    Js[np.ix_(ind, ind)] -= dJ

  return Fs, Js