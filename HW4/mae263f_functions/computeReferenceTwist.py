from .parallel_transport import parallel_transport
from .rotateAxisAngle import rotateAxisAngle
from .signedAngle import signedAngle

def computeReferenceTwist(a1e, a1f, t1, t2, refTwist = None):
    if refTwist is None:
      refTwist = 0
    P_a1e = parallel_transport(a1e, t1, t2)
    P_a1e_t = rotateAxisAngle(P_a1e, t2, refTwist)
    refTwist = refTwist + signedAngle(P_a1e_t, a1f, t2)
    return refTwist