import numpy as np
import scipy.special as sp  # For calculating Laguerre polinomials
import scipy.optimize as so
import ElectromagneticField as EMF
import Particle as PT
import LG_basis as LG

n = 12
z_foc         = 8.1487 / 0.127324
z_0 = 13.644641
tau = 2 * np.pi * n
t_foc = z_foc + np.pi*n                          # Center of time envelope
tmax = tau# t_foc + 0.5 * tau + tau           # Final moment of time
tmin = 0. # t_foc - 0.5 * tau - 0.1 * tau           # Initial moment of time
ntime = 10010                        # Number of time points
t_ntime = ntime

w0 = 10. # 15*np.pi # 2.0 /0.127324
zR = 0.5 * w0 ** 2
p = 0
l = 1
alpha = 0.
dz = 0.0000000000001

output_dir = "/home/bearlune/output_r0"

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
    z = z - z_foc
    r = [x, y, z]
    t = t - t_foc
    LG_base = LG.LG_basis(w0)
    return a0 * gt * np.real(LG_base.LG(r, t, p, l, 'x'))


def u(r):
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return a0 * Cpl * (w0/w)\
              * (np.sqrt(2) * r/w)**np.abs(l) * np.exp(-(r/w)**2) * sp.eval_genlaguerre(p, np.abs(l), 2*(r/w)**2)


def dudr(r):
    f = u(r)
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return f * (np.abs(l) / r - 2. * r / w ** 2) - \
           f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)


def ududr(r):
    f = u(r)
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    factor = 0
    if p == 0:
        return f**2 * (np.abs(l) / r - 2. * r / w ** 2)
    return f**2 * (np.abs(l) / r - 2. * r / w ** 2) - \
           factor*f*a0*4 * r / w ** 2 * Cpl * (w0 / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)


def ududz(r):
    f = u(r)
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    Cpl = np.sqrt(2.0 * np.math.factorial(int(p)) / (np.pi * np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    if p ==0:
        return z/zR**2 / (1. + z**2/zR**2) * f**2 * (2*r**2/w**2 - np.abs(l) - 1)
    return z/zR**2 / (1. + z**2/zR**2) * (f**2 * (2*r**2/w**2 - np.abs(l) - 1) - f * a0*Cpl*w0/w*(np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2) * (4*r**2/w**2))


def u_gradu_cyl(r):
    z_component = 0 #u(r)*(u(np.add(r, [0, 0, dz])) - u(r))/dz # non-paraxial
    return np.array((ududr(r), 0., z_component))

def ph(r, t):
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return t - t_foc - z - l*theta - (0.0 if z == 0 else r**2/(2*z*(1. + (zR/z)**2)))\
        + (2*p + np.abs(l) + 1) * np.arctan(z/zR) + alpha

def dphidr(r):
    x, y, z = r
    z = z - z_foc
    r = np.sqrt(x ** 2 + y ** 2)
    return - (0.0 if z == 0 else r/(z*(1. + (zR/z)**2)))

def gradph_cyl(r, t):
    x, y, z = r
    return np.array((dphidr(r), -l/np.sqrt(x0**2 + y0**2), -1.))


def Ph(r, t):
    x, y, z = r
    return (t - np.pi*n) * np.pi / tau


def p1(r0, t):
    #if 0 < t < tau:
    #    return -1 * dot_LP_LG_beam_x(r0, t)[0] - np.sin(2 * Ph(r0, t)) / (2 * n) * u(r0) * np.cos(ph(r0, t))
    return -1 * dot_LP_LG_beam_x(r0, t)[0]


def r1(r0, t):
    return -1 * LP_LG_beam_x(r0, t)[0]


def p2_cyl(r0, t):
    ugradu = u_gradu_cyl(r0)
    if t < 0:
        g2int = 0
    elif t > tau:
        g2int = 3./8.*tau
    else:
        g2int = 3./8.*t + 0.5*n*np.sin(2*Ph(r0, t)) + 1./16.*n*np.sin(4*Ph(r0, t))
    return -0.5*(ugradu*g2int + 0.5*ugradu*g(t)**2*np.sin(2*ph(r0, t)) + 0.5*u(r0)**2*g(t)**2*np.cos(2*ph(r0, t))*gradph_cyl(r0, t)) - np.cross(LP_LG_beam_x(r0, t)[0], LP_LG_beam_x(r0, t)[1])

def r2_cyl(r0, t):
    ugradu = u_gradu_cyl(r0)

    x0, y0, z0 = r0
    theta0 = np.arctan2(y0, x0)
    r0_2 = np.sqrt(x0**2 + y0**2)

    if t < 0:
        g2int = 0
    elif t > tau:
        g2int = 3./8.*tau
    else:
        g2int = 3./8.*t + 0.5*n*np.sin(2*Ph(r0, t)) + 1./16.*n*np.sin(4*Ph(r0, t))
    if t < 0:
        g2double_int = 0
    elif t > tau:
        g2double_int = 3. / 16. * tau**2
    else:
        g2double_int = 3. / 16. * t**2 - 0.5 * n**2 * (np.cos(2 * Ph(r0, t)) + 1.) - 1./32. * n**2 * (np.cos(4 * Ph(r0, t)) - 1.)
    EBx = 0.25*g(t)**2 * ((ududr(r0)*np.cos(2*ph(r0, t)) - u(r0)**2 * np.sin(2*ph(r0, t))*dphidr(r0))*np.cos(theta0) - l/r0_2*u(r0)**2 * np.sin(2*ph(r0, t))*np.sin(theta0)) - 0.5*l/r0_2 * u(r0)**2 * g2int * np.sin(theta0)
    EBy = 0.25*g(t)**2 * ((ududr(r0)*np.cos(2*ph(r0, t)) - u(r0)**2 * np.sin(2*ph(r0, t))*dphidr(r0))*np.sin(theta0) + l/r0_2*u(r0)**2 * np.sin(2*ph(r0, t))*np.cos(theta0)) + 0.5*l/r0_2 * u(r0)**2 * g2int * np.cos(theta0)
    EBz = 0.5*u(r0)**2*g2int - 0.25*u(r0)**2 * g(t)**2 * np.sin(2*ph(r0, t))
    return -0.5*(ugradu*g2double_int - 0.25*ugradu*g(t)**2*np.cos(2*ph(r0, t)) + 0.25*u(r0)**2*g(t)**2*np.sin(2*ph(r0, t))*gradph_cyl(r0, t)) - np.array((EBx*np.cos(theta0) + EBy*np.sin(theta0), -EBx*np.sin(theta0) + EBy*np.cos(theta0), EBz))


# ================================= Determine particles and fields =====================================================
emf = EMF.ElectromagneticField(LP_LG_beam_x)      # Particle moves in this field
arr_t = np.linspace(tmin - 2*2.*np.pi, tmax + 2*2.*np.pi, ntime)              # Array of time
t_arr_t = np.linspace(tmin - 2*2.*np.pi, tmax + 2*2.*np.pi, t_ntime)

# Initial conditions
z0 = 13.644641

nb_r = 10
r_min = 0.0001*0.5*w0 # w0 + 0.1*w0 #+ 0.1*w0 #7.4687 #w0*np.sqrt(np.abs(l)*0.5)
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
a0_min = -0.001
a0_max = -0.0000001
a0_space = np.linspace(a0_min, a0_max, 1)
for a0 in a0_space:
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
            pr_fin.append(pt.get_pr()[-1])
            ptheta_fin.append(pt.get_ptheta()[-1])
            px_fin.append(pt.get_px()[-1])
            py_fin.append(pt.get_py()[-1])
            pz_fin.append(pt.get_pz()[-1])
            Lz_fin.append(pt.get_Lz()[-1])
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
    with open(output_dir + "/pr.txt", "w") as f:
        for pr in pr_r0:
            f.write("%s\n" % pr)
    with open(output_dir + "/pz.txt", "w") as f:
        for pz in pz_r0:
            f.write("%s\n" % pz)
    with open(output_dir + "/Lz.txt", "w") as f:
        for Lz in Lz_r0:
            f.write("%s\n" % Lz)

# Here we plot averaged values for a particle
#a0 = -1.
#a0_tspace = np.linspace(a0_min, a0_max, 100)
#popt = so.curve_fit(f, np.abs(a0_space), Lz_arr)[0]
#print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
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
