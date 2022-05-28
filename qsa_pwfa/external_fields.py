import numpy as np


class GaussianBunchField:

    def __init__(self, n_b, sigma_R, sigma_Xi, ksi0):
        """
            exp( -0.5 * (ksi-ksi0)**2 / sigma_Xi**2 ) * exp( -0.5 * r**2 / sigma_R**2)
        """
        self.n_b = n_b
        self.sigma_R = sigma_R
        self.ksi0 = ksi0
        self.sigma_Xi = sigma_Xi
        self.N_sigma = 4.5

    def get_dAz_dr(self, r, ksi):
        val = self.n_b / r \
            * (np.abs(ksi-self.ksi0)<=self.N_sigma*self.sigma_Xi)  \
            * np.exp( -0.5 * (ksi-self.ksi0)**2 / self.sigma_Xi**2) \
            * self.sigma_R**2 * ( 1. - np.exp( -0.5 * r**2 / self.sigma_R**2 ) )

        return val
