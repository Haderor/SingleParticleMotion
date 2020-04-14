import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


class Particle:
    def __init__(self, r0_3, p0, F):
        # Initial values
        self.r0_3 = r0_3
        self.p0 = p0
        self.F = F  # Field
        self.arr_t = None
        # Values
        self.r_3 = None
        self.p = None

    # =================================== Get initial conditions and field =============================================
    def get_r0_3(self):
        return self.r0_3

    def get_p0(self):
        return self.p0

    def get_F(self):
        return self.F

    # =================================== Set initial conditions and field =============================================
    def set_r0_3(self, r0_3):
        self.r0_3 = r0_3

    def set_p0(self, p0):
        self.p0 = p0

    def set_F(self, F):
        self.F = F

    # ===================================== Obtaining trajectory =======================================================
    def __sys(self, t, X):
        x, y, z, px, py, pz = X
        p = np.array((px, py, pz))
        r = np.array((x, y, z))
        E, H = self.get_F().get_EH(r, t)
        v = p * (1. / np.sqrt(1. + np.dot(p, p)))

        return np.hstack((
            v,
            E + np.cross(v, H)
        ))

    # ========================================= Setting values =========================================================
    def set_trajectory(self, arr_t):
        self.arr_t = arr_t
        x = solve_ivp(self.__sys, [self.arr_t[0], self.arr_t[len(self.arr_t) - 1]], np.hstack((self.get_r0_3(), self.get_p0())), t_eval=self.arr_t, atol=1e-35, rtol=1e-35)
        self.r_3 = np.array([np.array((x.y[0][t], x.y[1][t], x.y[2][t])) for t in range(len(self.arr_t))])
        self.p = np.array([np.array((x.y[3][t], x.y[4][t], x.y[5][t])) for t in range(len(self.arr_t))])
        #x = odeint(self.__sys, np.hstack((self.get_r0_3(), self.get_p0())), self.arr_t)
        #self.r_3 = np.array([np.array((x[:, 0][t], x[:, 1][t], x[:, 2][t])) for t in range(len(self.arr_t))])
        #self.p = np.array([np.array((x[:, 3][t], x[:, 4][t], x[:, 5][t])) for t in range(len(self.arr_t))])

    # ========================================= Getting values =========================================================
    def get_t(self):
        return self.arr_t

    def get_r_3(self):
        return self.r_3

    def get_x(self):
        return np.array([self.get_r_3()[t][0] for t in range(len(self.arr_t))])

    def get_y(self):
        return np.array([self.get_r_3()[t][1] for t in range(len(self.arr_t))])

    def get_z(self):
        return np.array([self.get_r_3()[t][2] for t in range(len(self.arr_t))])

    def get_r(self):
        return np.sqrt(self.get_x()**2 + self.get_y()**2)

    def get_theta(self):
        return np.arctan2(self.get_y(), self.get_x())

    def get_p(self):
        return self.p

    def get_px(self):
        return np.array([self.get_p()[t][0] for t in range(len(self.arr_t))])

    def get_py(self):
        return np.array([self.get_p()[t][1] for t in range(len(self.arr_t))])

    def get_pz(self):
        return np.array([self.get_p()[t][2] for t in range(len(self.arr_t))])

    def get_pr(self):
        return self.get_px()*np.cos(self.get_theta()) + self.get_py()*np.sin(self.get_theta())

    def get_ptheta(self):
        return -self.get_px()*np.sin(self.get_theta()) + self.get_py()*np.cos(self.get_theta())

    def get_Lz(self):
        return self.get_r() * self.get_ptheta()

    def get_gamma(self):
        return np.sqrt(1. + self.get_px()**2 + self.get_py()**2 + self.get_pz()**2)

    def get_kineticE(self):
        return np.sqrt(1. + self.get_px() ** 2 + self.get_py() ** 2 + self.get_pz() ** 2) - 1.

    # ========================================== Plotters ==============================================================
    def plot(self, value, abscissa, ordinate):
        if value == 'r_3':
            if abscissa == 't':
                if ordinate == 'x':
                    plt.plot(self.get_t(), self.get_x())
                    plt.title('$x(t)$')
                if ordinate == 'y':
                    plt.plot(self.get_t(), self.get_y())
                    plt.title('$y(t)$')
                if ordinate == 'z':
                    plt.plot(self.get_t(), self.get_z())
                    plt.title('$z(t)$')
                if ordinate == 'r':
                    plt.plot(self.get_t(), self.get_r())
                    plt.title('$r(t)$')
                if ordinate == 'theta':
                    plt.plot(self.get_t(), self.get_theta())
                    plt.title('$\\theta(t)$')
                if ordinate == 'px':
                    plt.plot(self.get_t(), self.get_px())
                    plt.title('$p_x(t)$')
                if ordinate == 'py':
                    plt.plot(self.get_t(), self.get_py())
                    plt.title('$p_y(t)$')
                if ordinate == 'pz':
                    plt.plot(self.get_t(), self.get_pz())
                    plt.title('$p_z(t)$')
                if ordinate == 'pr':
                    plt.plot(self.get_t(), self.get_pr())
                    plt.title('$p_r(t)$')
                if ordinate == 'ptheta':
                    plt.plot(self.get_t(), self.get_ptheta())
                    plt.title('$p_\\theta(t)$')
                if ordinate == 'Lz':
                    plt.plot(self.get_t(), self.get_Lz())
                    plt.title('$L_z(t)$')
                if ordinate == 'E':
                    plt.plot(self.get_t(), self.get_gamma())
                    plt.title('$E(t)$')
                if ordinate == 'kineticE':
                    plt.plot(self.get_t(), self.get_kineticE())
                    plt.title('$kineticE(t)$')
                if ordinate == 'z+t':
                    plt.plot(self.get_t(), self.get_z(), label='z(t)')
                    plt.title('$z(t)$')
                if ordinate == 'r+t':
                    plt.plot(self.get_t(), self.get_r(), label='r(t)')
                    plt.title('$r(t)$')
                if ordinate == 'theta+t':
                    plt.plot(self.get_t(), self.get_theta(), label='theta(t)')
                    plt.title('$\\theta(t)$')
                if ordinate == 'pr+t':
                    plt.plot(self.get_t(), self.get_pr(), label='pr(t)')
                    plt.title('$pr(t)$')
            if abscissa == 'x':
                if ordinate == 'y':
                    plt.plot(self.get_x(), self.get_y())
                    plt.title('trajectory in (x, y)')
            plt.legend(loc='upper right')
            plt.xlabel(abscissa)
            plt.ylabel(ordinate)
            plt.show()
        if value == 'r_3' and abscissa == '' and ordinate == '':
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_title('Trajectory')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.plot3D(self.get_x(), self.get_y(), self.get_z())
            plt.show()
        if value == 'r_3+t' and abscissa == '' and ordinate == '':
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_title('Trajectory')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.plot3D(self.get_x(), self.get_y(), self.get_z())
            plt.show()

