import numpy as np # pyright: ignore[reportMissingImports]
import mae263f_functions as mf


that_boi = mf.crossMat(np.array([1,0,0])) @ np.array([0,1,0])

print(that_boi)

dF, dJ = mf.gradEb_hessEb(node0 = np.array([0,0,0]), node1 = np.array([1,0,0]), node2 = np.array([1,1,0]), l_k = 1, EI1 = 1)

