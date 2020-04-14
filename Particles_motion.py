import numpy as np
import matplotlib.pyplot as plt
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

w0 = 10.# 15*np.pi # 2.0 /0.127324
zR = 0.5 * w0 ** 2
p = 0
l = 2
alpha = 0.
#coeff_Rachel = 1. / ( 6/pi(w0**np.abs(l)) * (np.sqrt(0.5*np.abs(l)))**np.abs(l)*np.exp(-0.5*np.abs(l)))
#Apl = np.sqrt(2.0/(np.pi*np.math.factorial(np.abs(l)))) * np.pi * w0**(np.abs(l)+1) / (2.*2**(0.5*np.abs(l)))
#a0 = -1. #* coeff_Rachel * Apl
dz = 0.0000000000001

def f(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4


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


# ======================================================= Vladimir field ===============================================
def F(r):
    x, y, z = r
    r = np.sqrt(x ** 2 + y ** 2)
    w = w0 * np.sqrt(1. + (z / zR) ** 2)
    return a0*np.sqrt(np.math.factorial(int(p))/(np.math.factorial(p + np.abs(l)))) * (r**2/w**2)**(np.abs(l)/2.) * sp.eval_genlaguerre(p, np.abs(l), (r**2/w**2)) * np.exp(-(r**2/w**2)/2.)


def Vladimir_field(r, t):
    r3 = r
    x, y, z = r
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    w = w0*np.sqrt(1. + (z/zR)**2)

    Fv = w0/w * F(r3)
    dFvdr = Fv *(np.abs(l) - r**2/w0**2) / r
    Ev = Fv * np.cos(t - z - l * theta + alpha - z*r**2/(2*zR*w**2) + (2*p + np.abs(l) + 1)*np.arctan(z/zR))
    gt = g(t - z)

    Ex = gt * Ev
    Ey = 0.
    Ez = dFvdr * gt * np.cos(theta) * np.sin(t - z - l * theta + alpha - z*r**2/(2*zR*w**2) + (2*p + np.abs(l) + 1)*np.arctan(z/zR)) + l / r * Ex * np.sin(theta)

    Hx = -Ey
    Hy = Ex
    Hz = dFvdr * gt * np.sin(theta) * np.sin(t - z - l * theta + alpha - z*r**2/(2*zR*w**2) + (2*p + np.abs(l) + 1)*np.arctan(z/zR)) - l / r * Ex * np.cos(theta)

    E = Ex, Ey, Ez
    H = Hx, Hy, Hz
    return E, H


# ===================================== Linearly polrarized LG beam along x axis =======================================
def LP_LG_beam_x(r, t):
    x, y, z = r
    gt = g(t - z + z_0)
    z = z - z_foc
    r = [x, y, z]
    t = t - t_foc
    LG_base = LG.LG_basis(w0)
    return a0 * gt * np.real(LG_base.LG(r, t, p, l, 'x'))


def dot_LP_LG_beam_x(r, t):

    x, y, z = r
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    gt = g(t - z + z_0)
    z = z - z_foc

    Cpl = np.sqrt(2.0*np.math.factorial(int(p))/(np.pi*np.math.factorial(p + np.abs(l))))
    w = w0 * np.sqrt(1. + (z / zR)**2)
    phi = t + np.pi/2 - t_foc - z - l*theta - (0.0 if z == 0 else r**2/(2*z*(1. + (zR/z)**2)))\
        + (2*p + np.abs(l) + 1) * np.arctan(z/zR) + alpha
    u = Cpl * (w0/w) * (np.sqrt(2) * r/w)**np.abs(l) * np.exp(-(r/w)**2) * sp.eval_genlaguerre(p, np.abs(l), 2*(r/w)**2)

    LGpl = u * np.cos(phi)
    dudr = u * (np.abs(l) / r - 2. * r / w**2) -\
           4 * r / w**2 * Cpl * (1./w) * (np.sqrt(2) * r/w)**np.abs(l) * np.exp(-(r/w)**2) *\
           sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2*(r/w)**2)
    dphidr = - (0.0 if z == 0 else r/(z*(1. + (zR/z)**2)))
    diLGdr = -dudr * np.sin(phi) - LGpl * dphidr

    Ex = a0 * gt * LGpl
    Ey = 0.
    Ez = a0 * gt * (-diLGdr * np.cos(theta) + l / r * LGpl * np.sin(theta))

    Hx = - Ey
    Hy = Ex
    Hz = a0 * gt * (-diLGdr * np.sin(theta) - l / r * LGpl * np.cos(theta))

    E = Ex, Ey, Ez
    H = Hx, Hy, Hz
    return np.array((E, H))



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
a0_space = np.linspace(a0_min, a0_max, 10)
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

            '''
            plt.subplot(2, 3, 1)
            plt.plot(arr_t / (2*np.pi), pt.get_px(), '-k', label='px', linewidth=4)
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 3, 2)
            plt.plot(arr_t / (2*np.pi), pt.get_py(), '-k', linewidth=4, label='py')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 3, 3)
            plt.plot(arr_t / (2*np.pi), pt.get_pz(), '-k', linewidth=4, label='pz')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 3, 4)
            plt.plot(arr_t / (2*np.pi), pt.get_x(), '-k', linewidth=4, label='x')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 3, 5)
            plt.plot(arr_t / (2*np.pi), pt.get_y(), '-k', linewidth=4, label='y')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 3, 6)
            plt.plot(arr_t / (2*np.pi), pt.get_z(), '-k', linewidth=4, label='z')
            plt.legend(loc='best')
            plt.grid()

            #plt.show()
            plt.cla()
            plt.clf()
            '''

            '''
            lw = 1
            plt.subplot(2, 4, 1)
            plt.plot(arr_t / (2*np.pi), pt.get_pr(), '-k', linewidth=lw, label='pr')

            #te_pr = np.zeros(len(arr_t))
            #for i in range(len(arr_t)):
            #    if 0 <= arr_t[i] <= tau:
            #        te_pr[i] = u(r0_3) * np.cos(theta0) * 0.25 * (
            #                    2 * np.sin(ph(r0_3, arr_t[i])) + n / (n + 1.) * np.sin(2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + n / (
            #                        n - 1.) * np.sin(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) - 2 / (
            #                                n ** 2 - 1) * np.sin(-ph(r0_3, 0.)))
            #plt.plot(t_arr_t / (2 * np.pi), te_pr, label='exact theor')

            t_pr = [p1(r0_3, t)[0] * np.cos(theta0) + p1(r0_3, t)[1] * np.sin(theta0) + p2_cyl(r0_3, t)[0] for t in t_arr_t]
            plt.plot(t_arr_t / (2*np.pi), t_pr, label='theor')
            #plt.plot(arr_t / (2*np.pi), [-u(r_0)*np.cos(theta0) - 0.5*3./8.*t*ududr(r_0)*2 for t in arr_t], label='simple_theor')
            plt.legend(loc='best')
            plt.grid()


            plt.subplot(2, 4, 2)
            plt.plot(arr_t / (2*np.pi), pt.get_ptheta(), '-k', linewidth=lw, label='ptheta')

            #te_ptheta = np.zeros(len(arr_t))
            #for i in range(len(arr_t)):
            #    if 0 <= arr_t[i] <= tau:
            #        te_ptheta[i] = -u(r0_3) * np.sin(theta0) * 0.25 * (
            #                2 * np.sin(ph(r0_3, arr_t[i])) + n / (n + 1.) * np.sin(
            #            2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + n / (
            #                        n - 1.) * np.sin(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) - 2 / (
            #                        n ** 2 - 1) * np.sin(-ph(r0_3, 0.)))
            #    #elif arr_t[i] > tau:
            #    #    te_ptheta[i] = u(r0_3) * (u([x0, y0, z0 + dz]) - u([x0, y0, z0]))/dz * np.sin(theta0)**2 * 3./16. * tau *l / r0
            #plt.plot(t_arr_t / (2 * np.pi), te_ptheta, label='exact theor')

            t_ptheta = np.array([-p1(r0_3, t)[0] * np.sin(theta0) + p1(r0_3, t)[1] * np.cos(theta0) + p2_cyl(r0_3, t)[1] for t in t_arr_t])
            plt.plot(t_arr_t / (2 * np.pi), t_ptheta, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 4, 3)
            plt.plot(arr_t / (2*np.pi), pt.get_pz(), '-k', linewidth=lw, label='pz')

            #te_pz = np.zeros(len(arr_t))
            #for i in range(len(arr_t)):
            #    if 0 <= arr_t[i] <= tau:
            #        te_pz[i] = - u(r0_3) / r0 *(np.abs(l) - 2 * r0**2/(w0 * np.sqrt(1. + (z0 - z_foc)**2/zR**2))**2) * np.cos(theta0) * 0.25 * (
            #                2 * np.cos(ph(r0_3, arr_t[i])) + n / (n + 1.) * np.cos(
            #            2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + n / (
            #                        n - 1.) * np.cos(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + 2. / (
            #                        n ** 2 - 1) * np.cos(-ph(r0_3, 0.))) - te_ptheta[i] * l / r0 + te_pr[i] * dphidr(r0_3)
            #plt.plot(arr_t / (2 * np.pi), te_pz, label='exact theor')

            t_pz = np.array([p1(r0_3, t)[2] + p2_cyl(r0_3, t)[2] for t in t_arr_t])
            plt.plot(arr_t / (2 * np.pi), t_pz, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 4, 4)
            plt.plot(arr_t / (2*np.pi), pt.get_r(), '-k', linewidth=lw, label='r')

            #t_r = [r0]*len(arr_t)
            #for i in range(len(arr_t)):
            #    if 0 <= arr_t[i] <= tau:
            #        t_r[i] = r0 - u(r0_3) * np.cos(theta0) * 0.25 * (
            #                2 * np.cos(ph(r0_3, arr_t[i])) + (n / (n + 1.))**2 * np.cos(
            #            2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + (n / (
            #                        n - 1.))**2 * np.cos(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + 2 *(arr_t[i]*np.sin(-ph(r0_3, 0.)) + (3*n**2 - 1)/(n**2 - 1.)*np.cos(-ph(r0_3, 0.)))/(n**2 - 1.))
            #    elif arr_t[i] > tau:
            #        t_r[i] = r0 - u(r0_3) * np.cos(theta0) * 0.25 * (
            #                2 * np.cos(ph(r0_3, tau)) + (n / (n + 1.))**2 * np.cos(
            #            2 * Ph(r0_3, tau) + ph(r0_3, tau)) + (n / (
            #                        n - 1.))**2 * np.cos(-2 * Ph(r0_3, tau) + ph(r0_3, tau)) + 2 * (tau*np.sin(-ph(r0_3, 0.)) + (3*n**2 - 1)/(n**2 - 1.)*np.cos(-ph(r0_3, 0.)))/(n**2 - 1.))

            t_r = [r0 + r1(r0_3, t)[0]*np.cos(theta0) + r1(r0_3, t)[1]*np.sin(theta0) + r2_cyl(r0_3, t)[0] for t in arr_t]
            plt.plot(arr_t / (2 * np.pi), [r0 for t in arr_t], color='grey')
            plt.plot(arr_t / (2 * np.pi), t_r, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 4, 5)
            plt.plot(arr_t / (2*np.pi), pt.get_theta(), '-k', linewidth=lw, label='theta')


            t_theta = [theta0] * len(arr_t)
            for i in range(len(arr_t)):
                if 0 <= arr_t[i] <= tau:
                    t_theta[i] = theta0 + u(r0_3) / r0 * np.sin(theta0) * 0.25 * (
                            2 * np.cos(ph(r0_3, arr_t[i])) + (n / (n + 1.)) ** 2 * np.cos(
                        2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + (n / (
                            n - 1.)) ** 2 * np.cos(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) + 2 * (
                                        arr_t[i] * np.sin(-ph(r0_3, 0.)) + (3 * n ** 2 - 1) / (n ** 2 - 1.) * np.cos(
                                    -ph(r0_3, 0.))) / (n ** 2 - 1.))
                elif arr_t[i] > tau:
                    t_theta[i] = theta0 + u(r0_3) / r0 * np.sin(theta0) * 0.25 * (
                            2 * np.cos(ph(r0_3, tau)) + (n / (n + 1.)) ** 2 * np.cos(
                        2 * Ph(r0_3, tau) + ph(r0_3, tau)) + (n / (
                            n - 1.)) ** 2 * np.cos(-2 * Ph(r0_3, tau) + ph(r0_3, tau)) + 2 * (
                                        tau * np.sin(-ph(r0_3, 0.)) + (3 * n ** 2 - 1) / (n ** 2 - 1.) * np.cos(
                                    -ph(r0_3, 0.))) / (n ** 2 - 1.))


            t_theta = [theta0 - r1(r0_3, t)[0]*np.sin(theta0)/r0 + r1(r0_3, t)[1]*np.cos(theta0)/r0 + r2_cyl(r0_3, t)[2]/r0 for t in arr_t]
            plt.plot(arr_t / (2 * np.pi), [theta0 for t in arr_t], color='grey')
            plt.plot(arr_t / (2 * np.pi), t_theta, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 4, 6)
            plt.plot(arr_t / (2*np.pi), pt.get_z(), '-k', linewidth=lw, label='z')


            t_z = [z0]*len(arr_t)
            t_z = np.add(t_z,  - (t_theta - theta0) * l + (t_r - r0) * dphidr(r0_3))
            for i in range(len(arr_t)):
                if 0 <= arr_t[i] <= tau:
                    t_z[i] += -u(r0_3) / r0 * (np.abs(l) - 2 * r0 ** 2 / (w0 * np.sqrt(1. + (z0 - z_foc)**2/zR**2)) ** 2) * np.cos(theta0) * 0.25 * (
                            2 * np.sin(ph(r0_3, arr_t[i])) + (n / (n + 1.))**2 * (np.sin(
                        2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) - np.sin(-ph(r0_3, 0.))) + (n / (
                                    n - 1.))**2 * (np.sin(-2 * Ph(r0_3, arr_t[i]) + ph(r0_3, arr_t[i])) - np.sin(-ph(r0_3, 0.))) + 2 * (np.sin(-ph(r0_3, 0.)) + arr_t[i]*np.cos(-ph(r0_3, 0.))/(n**2 - 1)))
                elif arr_t[i] > tau:
                    t_z[i] += -u(r0_3) / r0 * (np.abs(l) - 2 * r0 ** 2 / (w0 * np.sqrt(1. + (z0 - z_foc)**2/zR**2)) ** 2) * np.cos(theta0) * 0.25 * (
                            2 * np.sin(ph(r0_3, tau)) + (n / (n + 1.))**2 * (np.sin(
                        2 * Ph(r0_3, tau) + ph(r0_3, tau)) - np.sin(-ph(r0_3, 0.))) + (n / (
                                    n - 1.))**2 * (np.sin(-2 * Ph(r0_3, tau) + ph(r0_3, tau)) - np.sin(-ph(r0_3, 0.))) + 2 * (np.sin(-ph(r0_3, 0.)) + tau*np.cos(-ph(r0_3, 0.))/(n**2 - 1)))


            t_z = [z0 + r1(r0_3, t)[2] + r2_cyl(r0_3, t)[2] for t in arr_t]
            plt.plot(arr_t / (2 * np.pi), [z0 for t in arr_t], color='grey')
            plt.plot(arr_t / (2 * np.pi), t_z, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.subplot(2, 4, 7)
            plt.plot(arr_t / (2*np.pi), pt.get_Lz(), '-k', linewidth=lw, label='Lz')
            t_Lz = r0 * t_ptheta
            plt.plot(t_arr_t / (2 * np.pi), t_Lz, label='theor')
            plt.legend(loc='best')
            plt.grid()

            plt.show()

            plt.cla()
            plt.clf()
            '''

            #tvx_fin.append(tvx1[len(arr_t) - 1] + tvx2[len(arr_t) - 1])
            pr_fin.append(pt.get_pr()[len(arr_t) - 1])
            ptheta_fin.append(pt.get_ptheta()[len(arr_t) - 1])
            px_fin.append(pt.get_px()[len(arr_t) - 1])
            py_fin.append(pt.get_py()[len(arr_t) - 1])
            pz_fin.append(pt.get_pz()[len(arr_t) - 1])
            Lz_fin.append(pt.get_Lz()[len(arr_t) - 1])
            #tLz_fin.append(-y0 * (tvx1[len(arr_t) - 1] + tvx2[len(arr_t) - 1]))

        # Plot values depend on theta
        plt.plot(theta_space, pr_fin, label='n')
        #plt.plot(theta_space, tvx_fin, label='t')
        #plt.plot(theta_space, [sum(pr_fin)/nb_theta for theta0 in theta_space], label='<$p_r$>, n')
        #plt.plot(theta_space, [sum(tvx_fin) / nb_theta for theta0 in theta_space], label='<$v_x$>, t')
        plt.title('$p_r(r_0 = $' + str(r0) + '$, \\theta_0, t = t_f)$')
        plt.legend(loc='best')
        plt.grid()
        #plt.show()

        plt.cla()
        plt.clf()
        plt.plot(theta_space, Lz_fin, label='n')
        #plt.plot(theta_space, tLz_fin, label='t')
        #plt.plot(theta_space, [sum(Lz_fin)/nb_theta for theta0 in theta_space], label='<$L_z$>, n')
        #plt.plot(theta_space, [sum(tLz_fin) / nb_theta for theta0 in theta_space], label='<$L_z$>, t')
        #plt.plot(theta_space, [np.pi * n * ududr([r0*np.cos(i), r0*np.sin(i), z0]) * np.sin(2*i)*0.25 * (n**2 + 1.) / (n**2 - 1.)**2 for i in theta_space], label='t for z-components')
        plt.title('$L_z(r_0 = $' + str(r0) + '$, \\theta_0, t = t_f)$')
        plt.legend(loc='best')
        #plt.show()


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
    flag = 1
    if flag == 1:
        plt.cla()
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.title('$t = t_f, a_0 = $' + str(a0) + '$, w_0 = $' + str(w0) + '$, p = $' + str(p) + '$, l = $' + str(l) + '$, n = $' + str(n) + ', temp shape = cos^2')
        plt.plot(r_space, pr_r0, label='$p_r$')
        plt.plot(np.linspace(r_min, r_max, 100), [-tau * 3./16. * ududr([r0, 0, z0]) for r0 in np.linspace(r_min, r_max, 100)], label='$p_r$ theor')
        plt.legend(loc='best')
        plt.grid()

        #plt.subplot(2, 2, 2)
        #plt.plot(r_space, ptheta_r0, label='$p_\\theta$')
        #plt.plot(np.linspace(r_min, r_max, 100), [l * ududz([r0, 0, z0]) * 3. / 16. * 0.5 * tau / r0 for r0 in np.linspace(r_min, r_max, 100)], label='t')
        #plt.legend(loc='best')
        #plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(r_space, Lz_r0, label='$L_z$')
        plt.plot(np.linspace(r_min, r_max, 100), [l * ududz([r0, 0, z0]) * 3./16. * 0.5 * tau for r0 in np.linspace(r_min, r_max, 100)], label='$L_z$ theor')
        plt.legend(loc='best')
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(r_space, pz_r0, label='$p_z$')
        plt.plot(np.linspace(r_min, r_max, 100), [(-3. / 16.) * tau * ududz([r0, 0, z0]) for r0 in np.linspace(r_min, r_max, 100)], label='$p_z$ theor')
        plt.legend(loc='best')
        plt.grid()

        plt.show()
        plt.cla()
        plt.clf()

# Here we plot averaged values for a particle
a0 = -1.
a0_tspace = np.linspace(a0_min, a0_max, 100)
popt = so.curve_fit(f, np.abs(a0_space), Lz_arr)[0]
print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
plt.cla()
plt.clf()
plt.plot(np.abs(a0_space), Lz_arr, label='Lz vs a0')
#plt.plot([0.1, 0.3, 1.0, 3.0, 10.0], [0.424, 3.98, 125, 6260, 131000], label='Vladimir calculation')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
#plt.plot(np.abs(a0_space), [0.25*tau * 3./8. * l * a0**2 * (u([r_min, 0, z0]))**2 for a0 in a0_space], label='theor')
r0_3 = [r_min, 0, z0]
plt.plot(np.abs(a0_tspace), [b0**2* l * ududz([r_min, 0, z0]) * 3./16. * 0.5 * tau for b0 in np.abs(a0_tspace)], label='theor')
plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
plt.plot(np.abs(a0_tspace), [a0**4*popt[4] for a0 in np.abs(a0_tspace)], label='~$a_0^4$')
plt.grid()
plt.legend()
plt.show()

plt.plot(np.abs(a0_space), np.sqrt(np.abs(Lz_arr)), label='sqrt(Lz) vs a0')
plt.grid()
plt.legend()
plt.show()

popt = so.curve_fit(f, np.abs(a0_space), pr_arr)[0]
print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
plt.cla()
plt.clf()
plt.plot(np.abs(a0_space), pr_arr, label='pr vs a0')
plt.plot(np.abs(a0_tspace), [b0**2 * (-3./16.) *tau * ududr([r_min, 0, z0]) for b0 in np.abs(a0_tspace)], label='theor')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
#plt.plot(np.abs(a0_space), [a0**4*popt[4] for a0 in np.abs(a0_space)], label='~$a_0^4$')
plt.legend()
plt.grid()
plt.show()

plt.plot(np.abs(a0_space), np.sqrt(np.abs(pr_arr)), label='sqrt(pr) vs a0')
plt.grid()
plt.legend()
plt.show()

popt = so.curve_fit(f, np.abs(a0_space), pz_arr)[0]
#plt.plot([0.1, 0.3, 1.0, 3.0, 10.0], [0.144, 2.08, 96.0, 5410, 112000], label='Vladimir calculation')
print('f(x) = ', popt[4], 'x^4 + ', popt[3], 'x^3 + ', popt[2], 'x^2 + ', popt[1], 'x + ', popt[0])
plt.plot(np.abs(a0_space), pz_arr, label='pz vs a0')
plt.plot(np.abs(a0_tspace), [b0**2 * (-3./16.) *tau * ududz([r_min, 0, z0]) for b0 in np.abs(a0_tspace)], label='theor')
#plt.plot(np.abs(a0_space), [f(a0, popt[0], popt[1], popt[2], popt[3], popt[4]) for a0 in np.abs(a0_space)], label='fit')
plt.plot(np.abs(a0_tspace), [a0*popt[1] for a0 in np.abs(a0_tspace)], label='~$a_0$')
plt.plot(np.abs(a0_tspace), [a0**2*popt[2] for a0 in np.abs(a0_tspace)], label='~$a_0^2$')
plt.plot(np.abs(a0_tspace), [a0**3*popt[3] for a0 in np.abs(a0_tspace)], label='~$a_0^3$')
plt.plot(np.abs(a0_space), [a0**4*popt[4] for a0 in np.abs(a0_space)], label='~$a_0^4$')
plt.grid()
plt.legend()
plt.show()

plt.plot(np.abs(a0_space), np.sqrt(np.abs(pz_arr)), label='sqrt(pz) vs a0')
plt.grid()
plt.legend()
plt.show()

plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(Lz_arr)), label='Lz vs a0')
plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(pz_arr)), label='pz vs a0')
plt.plot(np.log2(np.abs(a0_space)), np.log10(np.abs(pr_arr)), label='pr vs a0')
plt.grid()
plt.legend()
plt.show()

# Values per period