import numpy as np
from .inline_methods import *


class Simulation:

    def __init__(self, L_xi, N_xi, L_r, N_r, dens_func=None):
        self.init_grids(L_xi, N_xi, L_r, N_r, dens_func)
        self.allocate_data()

    def init_grids(self, L_xi, N_xi, L_r, N_r, dens_func):
        # iteration counter
        self.i_xi = 0

        # grid range and resolutions
        self.L_xi = L_xi
        self.N_xi = N_xi
        self.L_r = L_r
        self.N_r = N_r

        self.xi = L_xi / N_xi * np.arange(N_xi)
        self.dxi = self.xi[1] - self.xi[0]

        self.dr0 = L_r/N_r
        self.r0 = self.dr0 * np.arange(1,N_r+1)

        # rings volumes corresponding to the grid
        self.dV = self.dr0 * (self.r0 - 0.5*self.dr0)
        self.dV[0] = 0.5 * self.dr0**2

        # handle the non-uniform density
        if dens_func is not None:
            self.get_dPsi_dr_inline = get_dPsi_dr_inline
            self.dV *= dens_func(self.r0 - 0.5*self.dr0)
        else:
            self.get_dPsi_dr_inline = get_dPsi_dr_unif_inline

    def allocate_data(self):
        self.r = self.r0.copy()
        self.r_half = self.r0.copy()
        self.r_next = self.r0.copy()

        self.p_perp = np.zeros_like(self.r0)
        self.p_perp_next = np.zeros_like(self.r0)

        self.T = np.zeros_like(self.r0)
        self.v_z = np.zeros_like(self.r0)
        self.gamma = np.zeros_like(self.r0)

        self.dAz_dr = np.zeros_like(self.r0)
        self.dPsi_dr = np.zeros_like(self.r0)
        self.dp_perp_dxi = np.zeros_like(self.r0)
        self.dr_dxi = np.zeros_like(self.r0) 
        self.d2r_dxi2 = np.zeros_like(self.r0)
        
        self.Psi = np.zeros_like(self.r0)
        self.F = np.zeros_like(self.r0)
        

    def init_beam(self, n_b, R_b, ksi0, R_xi):
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

    def add_beam_field(self, xi_i):
        self.dAz_dr += self.gaussian_integrate(self.r, xi_i)/ self.r

    def get_motion_functions(self):
        self.T[:] = (1. + self.p_perp ** 2) / (1. + self.Psi) **2
        self.v_z[:] = (self.T - 1.) / (self.T + 1.)
        self.gamma[:] = 0.5 * (1. + self.Psi) * (self.T + 1.)

    def get_dAz_dr(self):
        self.dAz_dr = get_dAz_dr_inline(self.dAz_dr, self.r, self.dV, self.v_z)

    def get_dPsi_dr(self):
        self.dPsi_dr = self.get_dPsi_dr_inline(self.dPsi_dr, self.r, \
                                               self.r0, self.dV)

    def get_Psi(self, r_loc):
        self.Psi = get_psi_inline(self.Psi, r_loc, self.r0, self.dV)

    def get_dp_perp_dxi(self):
        self.F[:] = self.dPsi_dr + (1. - self.v_z) * self.dAz_dr
        
        self.dp_perp_dxi[:] = self.F / (1. - self.v_z)
    
    def get_dr_dxi(self):
        self.dr_dxi[:] = self.p_perp_next / (1. + self.Psi)
    
    def get_d2r_dxi2(self):
        # self.d2r_dxi2[:] = 
        pass
    
    def advance_xi(self, correct_Psi=True):

        self.get_Psi(self.r)
        self.get_motion_functions()
        self.get_dAz_dr()
        self.get_dPsi_dr()

        self.add_beam_field(self.xi[self.i_xi])

        self.get_dp_perp_dxi()

        self.p_perp_next[:] = self.p_perp + self.dxi * self.dp_perp_dxi

        if correct_Psi:
            self.get_dr_dxi()
            self.r_half[:] = self.r + 0.5 * self.dxi * self.dr_dxi
            fix_crossing_axis_r(self.r_half)
            self.get_Psi(self.r_half)

        self.get_dr_dxi()
        self.r_next[:] = self.r + self.dxi * self.dr_dxi
        fix_crossing_axis_rp(self.r_next, self.p_perp_next)

        self.r[:] = self.r_next
        self.p_perp[:] = self.p_perp_next

        self.i_xi += 1
