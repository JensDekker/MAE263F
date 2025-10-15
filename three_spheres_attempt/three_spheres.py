import numpy as np # pyright: ignore[reportMissingImports]
import mae263f_functions as mf

# Initial Positions
q1 = np.array([0, 0]) # Sphere 1 Position
q2 = np.array([0, 0]) # Sphere 2 Position
q3 = np.array([0, 0]) # Sphere 3 Position

q = np.array([q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]]) # 6x1 Position Vector

# Constants
g = 9.81 # m/s^2

# Sphere Radii
R1 = 0.25 # m
R2 = 0.5 # m 
R3 = 0.25 # m

# Rod Dimensions
delta_l = 0.5 # m
rod_radius = 0.05 # m
rod_area = np.pi * rod_radius**2 # m^2

# Material Properties
# Density
rho_metal = 2700 # kg/m^3
rho_fluid = 1000 # kg/m^3
rho = rho_metal - rho_fluid # kg/m^3

# Mechanical Properties
ElasticModulus = 70e9 # Pa
EA = ElasticModulus * rod_area # Axial Stiffness

# Fluid Vicosity
mu = 1e-3 # Pa.s

# Equations of Motion
# Mass Matrix
m1 = 4 / 3 * np.pi * R1**3 * rho_metal
m2 = 4 / 3 * np.pi * R2**3 * rho_metal
m3 = 4 / 3 * np.pi * R3**3 * rho_metal

M_1D = np.array([m1, m1, m2, m2, m3, m3]) # 1D Mass Matrix
M = np.diag(M_1D) # 6x6 Mass Matrix

# Mass Matrix Derivative
dM = np.zeros((6,6)) # 6x6 Mass Matrix Derivative


# Damping Matrix
c1 = 6 * np.pi * mu * R1
c2 = 6 * np.pi * mu * R2
c3 = 6 * np.pi * mu * R3

C_1D = np.array([c1, c1, c2, c2, c3, c3]) # 1D Damping Matrix
C = np.diag(C_1D) # 6x6 Damping Matrix

# Weight Vector
W1 = - 4 / 3 * np.pi * R1**3 * rho * g
W2 = - 4 / 3 * np.pi * R2**3 * rho * g
W3 = - 4 / 3 * np.pi * R3**3 * rho * g
W = np.array([0, W1, 0, W2, 0, W3]) # 6x1 Weight Vector

# Energy Gradient and Hessian


