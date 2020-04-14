import numpy as np
import matplotlib.pyplot as plt


class ElectromagneticField:
    def __init__(self, EH):     # EMF is prescribed by E and H as one function EH(t, x) that returns E, H - two vectors
        self.EH = EH

    # ========================================== Setters ===============================================================
    def set_EH(self, EH):
        self.EH = EH

    # ========================================== Getters ===============================================================
    def get_E(self, r, t):
        return self.EH(r, t)[0]

    def get_H(self, r, t):
        return self.EH(r, t)[1]

    def get_EH(self, r, t):
        return self.get_E(r, t), self.get_H(r, t)

    def get_I(self, r, t):
        return (np.dot(self.get_E(r, t), self.get_E(r, t)) + np.dot(self.get_H(r, t), self.get_H(r, t))) / (4. * np.pi)

    # ========================================== Plotters ==============================================================
    def plot(self, abscissa, ordinate, value, abscissa_min, abscissa_max, ordinate_min, ordinate_max, a, b):
        abscissa_n = 200
        ordinate_n = 200
        abscissa_arr = np.linspace(abscissa_min, abscissa_max, abscissa_n)
        ordinate_arr = np.linspace(ordinate_min, ordinate_max, ordinate_n)
        if value == 'Ex':
            if abscissa == 'x':
                if ordinate == 't':
                    value_arr = [self.get_E([x, a, b], t)[0] for t in ordinate_arr for x in abscissa_arr]
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet')
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
                if ordinate == 'y':
                    value_arr = [self.get_E([x, y, a], b)[0] for y in ordinate_arr[::-1] for x in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
        if value == 'Ey':
            if abscissa == 'x':
                if ordinate == 't':
                    value_arr = [self.get_E([x, a, b], t)[1] for t in ordinate_arr for x in abscissa_arr]
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet')
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
                if ordinate == 'y':
                    value_arr = [self.get_E([x, y, a], b)[1] for y in ordinate_arr[::-1] for x in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
        if value == 'Ez':
            if abscissa == 'x':
                if ordinate == 't':
                    value_arr = [self.get_E([x, a, b], t)[2] for t in ordinate_arr for x in abscissa_arr]
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet')
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
                if ordinate == 'y':
                    value_arr = [self.get_E([x, y, a], b)[2] for y in ordinate_arr[::-1] for x in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
        if value == 'Hz':
            if abscissa == 'x':
                if ordinate == 't':
                    value_arr = [self.get_H([x, a, b], t)[2] for t in ordinate_arr for x in abscissa_arr]
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet')
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
                if ordinate == 'y':
                    value_arr = [self.get_H([x, y, a], b)[2] for y in ordinate_arr[::-1] for x in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
        if value == 'I':
            if abscissa == 'x':
                if ordinate == 't':
                    value_arr = [self.get_I([x, a, b], t) for t in ordinate_arr[::-1] for x in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
                if ordinate == 'y':
                    value_arr = [self.get_I([x, y, a], b) for x in ordinate_arr[::-1] for y in abscissa_arr]
                    lim = max(np.abs(value_arr))
                    abscissa_arr, ordinate_arr = np.meshgrid(abscissa_arr, ordinate_arr)
                    plt.pcolormesh(abscissa_arr, ordinate_arr, np.reshape(value_arr, (abscissa_n, ordinate_n)),
                                   cmap='jet', vmin=-lim, vmax=lim)
                    plt.title(value)
                    plt.xlabel(abscissa)
                    plt.ylabel(ordinate)
                    plt.colorbar(shrink=.92)
                    plt.show()
