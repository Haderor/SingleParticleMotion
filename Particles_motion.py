import numpy as np
import scipy.special as sp  # For calculating Laguerre polinomials
import scipy.optimize as so
import ElectromagneticField as EMF
import Particle as PT
import LG_basis as LG

n = 12
#tau = 2 * np.pi * n
#tmax = tau # t_foc + 0.5 * tau + tau           # Final moment of time
#tmin = 0. # t_foc - 0.5 * tau - 0.1 * tau           # Initial moment of time
ntime = 10010                        # Number of time points
t_ntime = ntime

w0 = 10. # 15*np.pi # 2.0 /0.127324
zR = 0.5 * w0 ** 2
p = 0
l = 1
alpha = 0.
dz = 0.0000000000001
a0 = -0.01

z_0 = 0.001*zR

output_dir = "/home/bearlune/output_tau"

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
    gt = g(t - z + z_0)
    LG_base = LG.LG_basis(w0)
    return a0 * gt * np.real(LG_base.LG(r, t, p, l, 'x'))


def u(r):
    x, y, z = r
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return a0 * Cpl * (w0/w)\
              * (np.sqrt(2) * r/w)**np.abs(l) * np.exp(-(r/w)**2) * sp.eval_genlaguerre(p, np.abs(l), 2*(r/w)**2)


def dudr(r):
    f = u(r)
    x, y, z = r
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return f * (np.abs(l) / r - 2. * r / w ** 2) - \
           f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)


def ududr(r):
    f = u(r)
    x, y, z = r
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    if p == 0:
        return f**2 * (np.abs(l) / r - 2. * r / w ** 2)
    return f**2 * (np.abs(l) / r - 2. * r / w ** 2) - \
           f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)


def ududz(r):
    f = u(r)
    x, y, z = r
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    if p == 0:
        return z/zR**2 / (1. + z**2/zR**2) * f**2 * (2*r**2/w**2 - np.abs(l) - 1)
    return z/zR**2 / (1. + z**2/zR**2) * (f**2 * (2*r**2/w**2 - np.abs(l) - 1) - f * a0*Cpl*w0/w*(np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2) * (4*r**2/w**2))


def find_fin_moment(pt):
    arr_z = pt.get_z()
    for z in range(len(arr_z)):
        if arr_t[z] - arr_z[z] - z_0 > tau:
            return z
    print("Problem with final time!")
    return -1

# ================================= Determine particles and fields =====================================================
emf = EMF.ElectromagneticField(LP_LG_beam_x)      # Particle moves in this field
#arr_t = np.linspace(tmin - 2*2.*np.pi, tmax + 2*2.*np.pi, ntime)              # Array of time
#t_arr_t = np.linspace(tmin - 2*2.*np.pi, tmax + 2*2.*np.pi, t_ntime)

# Initial conditions
z0 = z_0

nb_r = 1
r_min = 0.5*w0 # w0 + 0.1*w0 #+ 0.1*w0 #7.4687 #w0*np.sqrt(np.abs(l)*0.5)
r_max = 3.*w0
r_space = np.linspace(r_min, r_max, nb_r)

nb_theta = 20
theta_min = -np.pi #np.pi #-139.4674 * np.pi / 180.
theta_max = np.pi - 2. * np.pi / nb_theta
theta_space = np.linspace(theta_min, theta_max, nb_theta)

Lz_arr = []
pr_arr = []
px_arr = []
py_arr = []
pz_arr = []
a0_min = -0.01
a0_max = -0.0000001
a0_space = np.linspace(a0_min, a0_max, 10)
n_min = 6
n_max = 15
n_space = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for n in n_space:
    tau = 2*np.pi * n
    t_min = 0.
    t_max = tau
    arr_t = np.linspace(t_min - 2*2.*np.pi, t_max + 2*2.*np.pi, ntime)
    Lz_r0 = []
    pr_r0 = []
    ptheta_r0 = []
    px_r0 = []
    py_r0 = []
    pz_r0 = []
    tLz_r0 = []
    tvx_r0 = []

    for r0 in r_space:
        print('r0 = ', r0)
        Lz_fin = []
        tLz_fin = []
        pr_fin = []
        ptheta_fin = []
        px_fin = []
        py_fin = []
        pz_fin = []
        tvx_fin = []
        for theta0 in theta_space:
            print('----', theta0)
            x0 = r0 * np.cos(theta0)
            y0 = r0 * np.sin(theta0)
            r0_3 = [x0, y0, z0]
            p0 = [0., 0., 0.]

            pt = PT.Particle(r0_3, p0, emf)                      # Moving particle

            # Setting calculated trajectory
            pt.set_trajectory(arr_t)

            #tvx_fin.append(tvx1[len(arr_t) - 1] + tvx2[len(arr_t) - 1])
            t_fin = find_fin_moment(pt)
            pr_fin.append(pt.get_pr()[t_fin])
            ptheta_fin.append(pt.get_ptheta()[t_fin])
            px_fin.append(pt.get_px()[t_fin])
            py_fin.append(pt.get_py()[t_fin])
            pz_fin.append(pt.get_pz()[t_fin])
            Lz_fin.append(pt.get_Lz()[t_fin])
            #tLz_fin.append(-y0 * (tvx1[len(arr_t) - 1] + tvx2[len(arr_t) - 1]))

        # Add values depend on r0
        pr_r0.append(sum(pr_fin) / nb_theta)
        ptheta_r0.append(sum(ptheta_fin) / nb_theta)
        px_r0.append(sum(px_fin) / nb_theta)
        py_r0.append(sum(py_fin) / nb_theta)
        pz_r0.append(sum(pz_fin) / nb_theta)
        Lz_r0.append(sum(Lz_fin) / nb_theta)
    Lz_arr.append(sum(Lz_r0)/nb_r)
    pr_arr.append(sum(pr_r0) / nb_r)
    px_arr.append(sum(px_r0) / nb_r)
    py_arr.append(sum(py_r0) / nb_r)
    pz_arr.append(sum(pz_r0) / nb_r)

    # Plot values depend on r0
    #with open(output_dir + "/pr.txt", "w") as f:
    #    for pr in pr_r0:
    #        f.write("%s\n" % pr)
    #with open(output_dir + "/pz.txt", "w") as f:
    #    for pz in pz_r0:
    #        f.write("%s\n" % pz)
    #with open(output_dir + "/Lz.txt", "w") as f:
    #    for Lz in Lz_r0:
    #        f.write("%s\n" % Lz)

with open(output_dir + "/pr.txt", "w") as f:
    for pr in pr_arr:
        f.write("%s\n" % pr)
with open(output_dir + "/pz.txt", "w") as f:
    for pz in pz_arr:
        f.write("%s\n" % pz)
with open(output_dir + "/Lz.txt", "w") as f:
    for Lz in Lz_arr:
        f.write("%s\n" % Lz)

# Here we plot averaged values for a particle
#a0 = -1.
#a0_tspace = np.linspace(a0_min, a0_max, 100)
#popt = so.curve_fit(f, np.abs(a0_space), Lz_arr)[0]
# print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
#plt.cla()
#plt.clf()
#plt.plot(np.abs(a0_space), Lz_arr, label='Lz vs a0')
#plt.plot([0.1, 0.3, 1.0, 3.0, 10.0], [0.424, 3.98, 125, 6260, 131000], label='Vladimir calculation')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
#plt.plot(np.abs(a0_space), [0.25*tau * 3./8. * l * a0**2 * (u([r_min, 0, z0]))**2 for a0 in a0_space], label='theor')
#r0_3 = [r_min, 0, z0]
#plt.plot(np.abs(a0_tspace), [b0**2* l * ududz([r_min, 0, z0]) * 3./16. * 0.5 * tau for b0 in np.abs(a0_tspace)], label='theor')
#plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
#plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
#plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
#plt.plot(np.abs(a0_tspace), [a0**4*popt[4] for a0 in np.abs(a0_tspace)], label='~$a_0^4$')
#plt.grid()
#plt.legend()
#plt.show()

#plt.plot(np.abs(a0_space), np.sqrt(np.abs(Lz_arr)), label='sqrt(Lz) vs a0')
#plt.grid()
#plt.legend()
#plt.show()

#popt = so.curve_fit(f, np.abs(a0_space), pr_arr)[0]
#print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
#plt.cla()
#plt.clf()
#plt.plot(np.abs(a0_space), pr_arr, label='pr vs a0')
#plt.plot(np.abs(a0_tspace), [b0**2 * (-3./16.) *tau * ududr([r_min, 0, z0]) for b0 in np.abs(a0_tspace)], label='theor')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
#plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
#plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
#plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
#plt.plot(np.abs(a0_space), [a0**4*popt[4] for a0 in np.abs(a0_space)], label='~$a_0^4$')
#plt.legend()
#plt.grid()
#plt.show()

#plt.plot(np.abs(a0_space), np.sqrt(np.abs(pr_arr)), label='sqrt(pr) vs a0')
#plt.grid()
#plt.legend()
#plt.show()

#popt = so.curve_fit(f, np.abs(a0_space), pz_arr)[0]
#plt.plot([0.1, 0.3, 1.0, 3.0, 10.0], [0.144, 2.08, 96.0, 5410, 112000], label='Vladimir calculation')
#print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
#plt.plot(np.abs(a0_space), pz_arr, label='pz vs a0')
#plt.plot(np.abs(a0_tspace), [b0**2 * (-3./16.) *tau * ududz([r_min, 0, z0]) for b0 in np.abs(a0_tspace)], label='theor')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
#plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
#plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
#plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
#plt.plot(np.abs(a0_space), [a0**4*popt[4] for a0 in np.abs(a0_space)], label='~$a_0^4$')
#plt.grid()
#plt.legend()
#plt.show()

#plt.plot(np.abs(a0_space), np.sqrt(np.abs(pz_arr)), label='sqrt(pz) vs a0')
#plt.grid()
#plt.legend()
#plt.show()

#plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(Lz_arr)), label='Lz vs a0')
#plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(pz_arr)), label='pz vs a0')
#plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(pr_arr)), label='pr vs a0')
#plt.grid()
#plt.legend()
#plt.show()

# Values per period
