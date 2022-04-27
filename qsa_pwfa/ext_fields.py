import numpy as np


class GaussianBeam:

    def __init__(self, n_b, R_b, ksi0, R_xi):
        """"
            exp( -0.5 * (ksi-ksi0)**2 / R_xi**2 ) * exp( -0.5 * r**2 / R_b**2)
        """
        self.n_b = n_b
        self.R_b = R_b
        self.ksi0 = ksi0
        self.R_xi = R_xi

    def get_dAz_dr(self, r, ksi):
        val = self.n_b / r * \
            np.exp( -0.5 * (ksi-self.ksi0)**2 / self.R_xi**2) * \
            self.R_b**2 * ( 1. - np.exp( -0.5 * r**2 / self.R_b**2 ) )

        return val