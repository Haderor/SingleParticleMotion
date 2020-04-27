import sys
from mpi4py import MPI
import numpy as np
import scipy.special as sp  # For calculating Laguerre polinomials
import ElectromagneticField as EMF
import Particle as PT
import LG_basis as LG

n = 12
tau = 2 * np.pi * n
tmax = tau # t_foc + 0.5 * tau + tau           # Final moment of time
tmin = 0. # t_foc - 0.5 * tau - 0.1 * tau           # Initial moment of time
ntime = 10010                        # Number of time points
t_ntime = ntime

a0 = 0.001
w0 = 10.
zR = 0.5 * w0 ** 2
p = 0
l = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = 16    # Number of running processes; number of particles on r0 is 2 * nb_proc
nb_r0 = 20      # Number of particles on the distance r0

def g(t):
    if tau == 0.0:
        return 0.0
    if tau == np.inf:
        return 1.0
    if t >= tau:
        return 0.0
    if t <= 0.:
        return 0.0
    return np.cos((t - np.pi*n) * np.pi / tau)**2


# ===================================== Linearly polrarized LG beam along x axis =======================================
def LP_LG_beam_x(r, t):
    x, y, z = r
    gt = g(t - z)
    r = [x, y, z]
    LG_base = LG.LG_basis(w0)
    return a0 * gt * np.real(LG_base.LG(r, t, p, l, 'x'))


# ================================= Determine particles and fields =====================================================
emf = EMF.ElectromagneticField(LP_LG_beam_x)      # Particle moves in this field
arr_t = np.linspace(tmin - 2*2.*np.pi, tmax + 2*2.*np.pi, ntime)              # Array of time

# Initial conditions
z0 = 0.
theta01 = -np.pi + np.pi / nb_proc * rank
theta02 = np.pi / nb_proc * rank
r0 = 0.0001 * w0 + 3. * w0 / nb_r0 * float(sys.argv[1])

x01 = r0 * np.cos(theta01)
x02 = r0 * np.cos(theta02)
y01 = r0 * np.sin(theta01)
y02 = r0 * np.sin(theta02)
r0_31 = [x01, y01, z0]
r0_32 = [x02, y02, z0]
p0 = [0., 0., 0.]

pt1 = PT.Particle(r0_31, p0, emf)                      # Moving particle
pt2 = PT.Particle(r0_32, p0, emf)

# Setting calculated trajectory
pt1.set_trajectory(arr_t)
pt2.set_trajectory(arr_t)
pr1 = pt1.get_pr()[-1]
pr2 = pt2.get_pr()[-1]

# -pi <= theta < 0
with open("/home/bearlune/OAM_transfer/Single_particle_effects/output/output1_" + str(sys.argv[1]) + "_" + str(rank) + ".txt", "w+") as f:
    f.write("%s\n" % pr1)

# 0 <= theta < pi
with open("/home/bearlune/OAM_transfer/Single_particle_effects/output/output2_" + str(sys.argv[1]) + "_" + str(rank) + ".txt", "w+") as f:
    f.write("%s\n" % pr2)
