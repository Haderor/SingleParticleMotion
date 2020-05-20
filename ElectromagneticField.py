import numpy as np

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
