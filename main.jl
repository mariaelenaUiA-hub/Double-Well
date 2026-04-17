using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), "")
qw = pyimport("quantum_well")


## Fuori dal Loop di training :

"""
1) Computation of the DVR grid and then,the T2D cinetic energy matrix; 
2) Computation of the Coulomb Kernel
3) Compuitation of Slater Basis  -- quelle ordinate dentro RL perchè dipåendono dal V

"""

Nx, Ny   = 20, 20
x0, xL   = -1.0,  1.0
y0, yL   = -0.5,  0.5
kappa    = 2326.0    # Coulomb strength (Mott-Hubbard crossover, U/t ~ 1)
epsilon  = 0.01 
Nstates  = 40   
N_CI_COMPUTE  = 200   # raised from 40; covers L0-L4 x R0-R4



M_LOC     = 12     # localisation subspace size (try 8, 12, 16, ...)
X_CUT     = 0.0    # dividing surface between wells (barrier centre)
SMOOTH_PL = True
SIGMA_PL  = 0.03      # SP orbitals to retain


#DVR
x_grid, y_grid, w_x, w_y, T2D = qw.build_2d_dvr(Nx, Ny, x0, xL, y0, yL)

#Kernel Coulomb
K = qw.precompute_coulomb_kernel(x_grid, y_grid, w_x, w_y, kappa, epsilon)

old_basis    = qw.build_slater_basis(Nstates)


##


