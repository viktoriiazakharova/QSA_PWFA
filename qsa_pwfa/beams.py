import numpy as np


class GaussianBeam:

    def __init__(self, n_b, R_b, ksi0, R_xi):
        self.n_b = n_b
        self.R_b = R_b
        self.ksi0 = ksi0
        self.R_xi = R_xi

    def gaussian_beam(self, r, ksi):
        """
        Gaussian beam density distribution
        """
        val = self.n_b * \
            np.exp( -0.5 * (ksi-self.ksi0)**2 / self.R_xi**2 ) * \
            np.exp( -0.5 * r**2 / self.R_b**2)

        return val

    def gaussian_integrate(self, r, ksi):
        """
        Gaussian beam density distribution integrated over `r`
        """
        val = self.n_b * \
            np.exp( -0.5 * (ksi-self.ksi0)**2 / self.R_xi**2) * \
            self.R_b**2 * ( 1. - np.exp( -0.5 * r**2 / self.R_b**2 ) )

        return val


