import numpy as np


class GaussianBunchField:

    def __init__(self, n_b, sigma_r, sigma_xi, xi_0=None, truncate_factor=4.0):
        """
            exp( -0.5 * (xi-xi0)**2 / sigma_xi**2 ) * exp( -0.5 * r**2 / sigma_r**2)
        """
        self.n_b = n_b
        self.sigma_r = sigma_r
        self.sigma_xi = sigma_xi
        
        if xi_0 is not None:
            self.xi_0 = xi_0
        else:
            self.xi_0 = truncate_factor * sigma_xi 
            
        self.truncate_factor = truncate_factor

    def get_dAz_dr(self, r, xi):
        val = self.n_b / r \
            * (np.abs(xi-self.xi_0) <= self.truncate_factor*self.sigma_xi) \
            * np.exp( -0.5 * (xi-self.xi_0)**2 / self.sigma_xi**2) \
            * self.sigma_r**2 * ( 1. - np.exp( -0.5 * r**2 / self.sigma_r**2 ) )

        return val
