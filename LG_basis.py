import numpy as np
import scipy.special as sp


# Represents p, l Laguerre-Gaussian mode
class LG_basis:
    def __init__(self, w0, omega=1):
        self.omega = omega                       # Laser frequency
        self.k = self.omega                      # Wave number. We assume speed of light c = 1
        self.w0 = w0                             # Beam waist
        self.zR = 0.5 * self.k * self.w0 ** 2    # Rayleigh range

    # Phase of the mode
    def phi(self, r, t, p, l):
        x, y, z = r
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return self.omega*t - self.k*z - l*theta - (0.0 if z == 0 else self.k*r ** 2 / (2 * z * (1. + (self.zR / z) ** 2))) + (2 * p + np.abs(l) + 1) * np.arctan(z / self.zR)

    # Beam width
    def w(self, z):
        return self.w0 * np.sqrt(1. + (z / self.zR) ** 2)

    # Normalization constant. Integral over cross-section gives w0**2
    def C(self, p, l):
        return np.sqrt(2.0*np.math.factorial(int(p))/(np.pi*np.math.factorial(p + np.abs(l))))

    # Amplitude
    def u(self, r, p, l):
        x, y, z = r
        r = np.sqrt(x ** 2 + y ** 2)
        w = self.w(z)
        return self.C(p, l) * \
               (self.w0 / w) * \
               (np.sqrt(2.) * r / w) ** np.abs(l) * \
               np.exp(-(r / w) ** 2) * \
               sp.eval_genlaguerre(p, np.abs(l), 2 * (r / w) ** 2)

    # LG mode corresponding p, l and directed along axis x or y
    def LG(self, r, t, p, l, axis):
        x, y, z = r
        u = self.u(r, p, l)  # Amplitude
        phi = self.phi(r, t, p, l)  # Phase

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        w = self.w(z)                      # Beam waste
        LGpl = u * np.exp(1j*phi)          # Linearly polarized mode
        dudr = u * (np.abs(l) / r - 2. * r / w ** 2) - \
               4 * r / w ** 2 * self.C(p, l) * (1. / w) * (np.sqrt(2) * r / w) ** np.abs(l) * np.exp(-(r / w) ** 2) * \
               sp.eval_genlaguerre(p - 1, np.abs(l) + 1, 2 * (r / w) ** 2)          # Derivative of amplitude respect r
        dphidr = - (0.0 if z == 0 else self.k*r / (z * (1. + (self.zR / z) ** 2)))  # Derivative of phase respect r
        dLGpldr = dudr * np.exp(1j*phi) + LGpl * 1j*dphidr                          # Derivative of the mode respect r

        Ex = 0.
        Ey = 0.
        dExdr = 0.
        dEydr = 0.
        if axis == 'x':        # Polarization along x axis
            Ex = LGpl
            Ey = 0.
            dExdr = dLGpldr  # Derivative of Ex respect r
            dEydr = 0.       # Derivative of Ey respect r
        elif axis == 'y':    # Polarization along y axis
            Ex = 0.
            Ey = LGpl
            dExdr = 0.       # Derivative of Ex respect r
            dEydr = dLGpldr  # Derivative of Ey respect r

        Er = Ex*np.cos(theta) + Ey*np.sin(theta)
        Etheta = -Ex*np.sin(theta) + Ey*np.cos(theta)
        dErdr = dExdr * np.cos(theta) + dEydr * np.sin(theta)
        dEthetadr = -dExdr * np.sin(theta) + dEydr * np.cos(theta)

        # Determine components of electromagnetic field Ez and H from Maxwell's equations
        Ez = (-1j*dErdr - l / r * Etheta) / self.k
        Hx = - Ey
        Hy = Ex
        Hz = (1j*dEthetadr - l / r * Er) / self.k

        E = Ex, Ey, Ez
        H = Hx, Hy, Hz
        return np.array((E, H))
