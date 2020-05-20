import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

nb_r0 = 20
w0 = 10.
zR = 0.5 * w0**2
a0 = 0.001
p = 0
l = 1
z0 = 0.
r0_min = 0.0001 * w0
r0_max = 3. * w0

nb_periods = 12
tau = 2 * np.pi * nb_periods

# ===================================== Reading from file =========================================
arr_pr = np.array([])
arr_pz = np.array([])
arr_Lz = np.array([])
arr_r0_pr = np.array([])
arr_r0_pz = np.array([])
arr_r0_Lz = np.array([])
arr_num = np.array([])
path_to_file = "output_r0/"

with open(path_to_file + "pr.txt", "r") as f:
    for line in f:
        arr_num = np.append(arr_num, float(line.split()[0]))
        arr_pr = np.append(arr_pr, float(line.split()[1]))
arr_r0_pr = r0_min + np.multiply(arr_num, r0_max / nb_r0)

arr_num = np.array([])
with open(path_to_file + "pz.txt", "r") as f:
    for line in f:
        arr_num = np.append(arr_num, float(line.split()[0]))
        arr_pz = np.append(arr_pz, float(line.split()[1]))
arr_r0_pz = r0_min + np.multiply(arr_num, r0_max / nb_r0)

arr_num = np.array([])
with open(path_to_file + "Lz.txt", "r") as f:
    for line in f:
        arr_num = np.append(arr_num, float(line.split()[0]))
        arr_Lz = np.append(arr_Lz, float(line.split()[1]))
arr_r0_Lz = r0_min + np.multiply(arr_num, r0_max / nb_r0)

# ===================================== Functions for analytical comparison  =========================================
def u(r, z):
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return a0 * Cpl * (w0/w) * (np.sqrt(2) * r/w)**np.abs(l) * np.exp(-(r/w)**2) * sp.eval_genlaguerre(p, np.abs(l), 2*(r/w)**2)

def dudr(r, z):
    f = u(r, z)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return f * (np.abs(l) / r - 2. * r / w ** 2) - \
           f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)

def ududr(r, z):
    f = u(r, z)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    if p == 0:
        return f**2 * (np.abs(l) / r - 2. * r / w ** 2)
    return f**2 * (np.abs(l) / r - 2. * r / w ** 2) - \
           f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)

def ududz(r, z):
    f = u(r, z)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    if p == 0:
        return z/zR**2 / (1. + z**2/zR**2) * f**2 * (2*r**2/w**2 - np.abs(l) - 1)
    return z/zR**2 / (1. + z**2/zR**2) * (f**2 * (2*r**2/w**2 - np.abs(l) - 1) - f * a0*Cpl*w0/w*(np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2) * (4*r**2/w**2))

# ========================================================= Plotting figures ===============================================
arr_r0_analytical = np.linspace(r0_min, r0_max, 1000)
plt.plot(arr_r0_pr, arr_pr, label = "$p_r$")
plt.plot(arr_r0_analytical, [-tau * 3./16. * ududr(r0, z0) for r0 in arr_r0_analytical], label = "$p_r$ analytic")
plt.legend(loc = "best")
plt.grid()
plt.show()

plt.plot(arr_r0_pz, arr_pz, label = "$p_z$")
plt.plot(arr_r0_analytical, [l * ududz(r0, z0) * 3./16. * 0.5 * tau for r0 in arr_r0_analytical],  label = "$p_z$ analytic")
plt.legend(loc = "best")
plt.grid()
plt.show()

plt.plot(arr_r0_Lz, arr_Lz, label = "$L_z$")
plt.plot(arr_r0_analytical, [(-3. / 16.) * tau * ududz(r0, z0) for r0 in arr_r0_analytical],  label = "$L_z$ analytic")
plt.legend(loc = "best")
plt.grid()
plt.show()
