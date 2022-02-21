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

        self.p_perp = np.zeros_like(self.r0)

        self.T = np.zeros_like(self.r0)
        self.v_z = np.zeros_like(self.r0)

        self.dAz_dr = np.zeros_like(self.r0)
        self.dAz_dr_beam  = np.zeros_like(self.r0)
        self.dPsi_dr = np.zeros_like(self.r0)

        self.dr_dxi = np.zeros_like(self.r0)

        self.d2r_dxi2 = np.zeros_like(self.r0)
        self.dAr_dxi = np.zeros_like(self.r0)
        self.dpsi_dxi = np.zeros_like(self.r0)

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

    def get_beam_field(self, xi_i):
        self.dAz_dr_beam = self.gaussian_integrate(self.r, xi_i)/ self.r

    def add_beam_field(self):
        self.dAz_dr += self.dAz_dr_beam

    def get_motion_functions(self, p_perp):
        self.T[:] = (1. + p_perp ** 2) / (1. + self.Psi) ** 2
        self.v_z[:] = (self.T - 1.) / (self.T + 1.)

    def get_dAz_dr(self):
        self.dAz_dr = get_dAz_dr_inline(self.dAz_dr, self.r, self.dV, self.v_z)

    def get_dPsi_dr(self):
        self.dPsi_dr = self.get_dPsi_dr_inline(self.dPsi_dr, self.r, \
                                               self.r0, self.dV)

    def get_Psi(self, r_loc):
        self.Psi = get_psi_inline(self.Psi, r_loc, self.r0, self.dV)

##############

    def get_force_reduced(self):
        self.F[:] = self.dPsi_dr + (1. - self.v_z) * self.dAz_dr

    def get_force_full(self):
        self.F += (1. - self.v_z) * self.dAr_dxi

##############

    def get_p_perp(self):
        self.p_perp[:] = self.dr_dxi * (1. + self.Psi)

    def get_d2r_dxi2(self):
        self.d2r_dxi2[:] = ( self.F / (1. - self.v_z) \
        - self.dpsi_dxi * self.dr_dxi ) / (1 + self.Psi)

    def get_dAr_dxi(self):
        self.dAr_dxi = get_dAr_dxi_inline(self.dAr_dxi, self.r,
                                          self.dr_dxi, self.d2r_dxi2, self.dV)

    def get_dpsi_dxi(self):
        self.dpsi_dxi = get_dpsi_dxi_inline(self.dpsi_dxi, self.r, self.r0,
                                            self.dr_dxi, self.dV)

    def advance_xi(self, mode_simple=True):
        self.get_dAr_dxi()

        self.r[:] = self.r + self.dxi * self.dr_dxi
        self.dr_dxi[:] = self.dr_dxi +  self.d2r_dxi2 * self.dxi

        self.get_Psi(self.r)
        self.get_dPsi_dr()
        self.get_dpsi_dxi()
        self.get_dAz_dr()
        self.get_beam_field(self.xi[self.i_xi])
        self.add_beam_field()
        self.get_p_perp()
        self.get_motion_functions(self.p_perp)

        self.get_force_reduced()
        self.get_force_full()

        self.get_d2r_dxi2()
        self.get_dAr_dxi()
        self.get_force_reduced()
        self.get_force_full()

        self.get_d2r_dxi2()
        fix_crossing_axis_rp(self.r, self.p_perp)
        self.i_xi += 1
