import numpy as np
from .inline_methods import *


class BaseSpecie:

    def init_r_grid(self, L_r, N_r, r_grid_user):

        if (L_r is not None) and (N_r is not None):
            self.L_r = L_r
            self.N_r = N_r
            self.r0 = L_r / N_r * np.arange(1, N_r+1)
            self.dr0 = np.gradient(self.r0)

            self.r0 -= 0.5*self.dr0
            self.dQ = self.dr0 * (self.r0 - 0.5*self.dr0)
            self.dQ[0] = 0.125 * self.dr0[0]**2

            self.rmax = self.r0.max()

        elif r_grid_user is not None:
            self.r0 = r_grid_user.copy()
            self.rmax = self.r0.max()
            self.L_r = self.rmax
            self.N_r = self.r0.size
            self.dr0 = np.gradient(self.r0)
            #np.r_[self.r0[0], self.r0[1:] - self.r0[:-1]]
            self.dQ = self.dr0 * (self.r0 - 0.5*self.dr0)
            self.dQ[0] = 0.5 * self.dr0[0]**2
        else:
            print('need to define the grid')

    def allocate_data(self):
        self.r = self.r0.copy()
        self.dr_dxi = np.zeros_like(self.r0)
        self.d2r_dxi2 = np.zeros_like(self.r0)
        self.d2r_dxi2_prev = np.zeros_like(self.r0)

        self.T = np.zeros_like(self.r0)
        self.v_z = np.zeros_like(self.r0)

        self.dAr_dxi = np.zeros_like(self.r0)
        self.dAz_dr = np.zeros_like(self.r0)

        self.Psi = np.zeros_like(self.r0)
        self.dPsi_dr = np.zeros_like(self.r0)
        self.dPsi_dxi = np.zeros_like(self.r0)

        self.F = np.zeros_like(self.r0)
        self.F_part = np.zeros_like(self.r0)

    def reinit(self):
        self.T[:] = 0.0
        self.v_z[:] = 0.0

        self.dAr_dxi[:] = 0.0
        self.dAz_dr[:] = 0.0

        self.Psi[:] = 0.0
        self.dPsi_dr[:] = 0.0
        self.dPsi_dxi[:] = 0.0

        self.F[:] = 0.0
        self.F_part[:] = 0.0


class PlasmaMethods:

    def get_Density(self, source_specie):
        weights = source_specie.dQ / (1 - source_specie.v_z)
        self.Density = get_Density_inline(self.Density, self.r0, self.dr0,
                                          source_specie.r, weights)
        self.Density /= self.dQ

    def get_dAz_dr(self, source_specie):
        self.dAz_dr = get_dAz_dr_inline(self.dAz_dr, self.r,
                                        source_specie.r,
                                        source_specie.v_z,
                                        source_specie.dQ)

    def get_Psi(self, source_specie):
        self.Psi = get_Psi_inline(self.Psi, self.r,
                                  source_specie.r,
                                  source_specie.r0,
                                  source_specie.dQ)

    def get_dPsi_dr(self, source_specie):
        if source_specie.type == "NeutralUniformPlasma":
            self.dPsi_dr = get_dPsi_dr_unif_inline(self.dPsi_dr, self.r,
                                                   source_specie.n_p,
                                                   source_specie.r,
                                                   source_specie.r0,
                                                   source_specie.dQ)
        elif source_specie.type == "NeutralNoneUniformPlasma":
            self.dPsi_dr = get_dPsi_dr_inline(self.dPsi_dr, self.r,
                                              source_specie.r,
                                              source_specie.r0,
                                              source_specie.dQ)

    def get_dPsi_dxi(self, source_specie):
        self.dPsi_dxi = get_dPsi_dxi_inline(self.dPsi_dxi, self.r,
                                            source_specie.r,
                                            source_specie.r0,
                                            source_specie.dr_dxi,
                                            source_specie.dQ)

    def get_vz(self):
        self.T[:] = (1. + (self.dr_dxi * (1. + self.Psi)) ** 2) / \
                    (1. + self.Psi) ** 2
        self.v_z[:] = (self.T - 1.) / (self.T + 1.)

    def get_dAr_dxi(self, source_specie):
        self.dAr_dxi = get_dAr_dxi_inline(self.dAr_dxi, self.r,
                                          source_specie.r,
                                          source_specie.dr_dxi,
                                          source_specie.d2r_dxi2,
                                          source_specie.dQ)

    def get_force_reduced(self):
        self.F_part[:] = self.dPsi_dr + (1. - self.v_z) * self.dAz_dr

    def get_force_full(self):
        self.F[:] = self.F_part + (1. - self.v_z) * self.dAr_dxi
        if self.particle_boundary == 1:
            self.F *= ( self.r<=self.rmax )

    def get_d2r_dxi2(self):
        self.d2r_dxi2[:] = ( self.F / (1. - self.v_z) \
            - self.dPsi_dxi * self.dr_dxi ) / (1 + self.Psi)

    def advance_motion(self, dxi):
            self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
            self.r += self.dr_dxi * dxi
            self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
            fix_crossing_axis_rp(self.r, self.dr_dxi)

class NeutralUniformPlasma(BaseSpecie, PlasmaMethods):

    def __init__(self, L_r=None, N_r=None, r_grid_user=None, n_p=1,
                 particle_boundary=1):

        self.type = "NeutralUniformPlasma"

        self.particle_boundary = particle_boundary
        self.n_p = n_p

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= n_p

        self.allocate_data()

class NeutralNoneUniformPlasma(BaseSpecie, PlasmaMethods):

    def __init__(self, dens_func, L_r=None, N_r=None, r_grid_user=None,
                 particle_boundary=0):

        self.type = "NeutralNoneUniformPlasma"

        self.particle_boundary = particle_boundary
        self.dens_func = dens_func

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= dens_func(self.r0)  #- 0.5*self.dr0)

        self.allocate_data()

class Grid(BaseSpecie, PlasmaMethods):

    def __init__(self, L_r=None, N_r=None, r_grid_user=None):

        self.particle_boundary = 0
        self.type = "Grid"
        self.init_r_grid(L_r, N_r, r_grid_user)

    def init_data(self, fields):
        self.r = self.r0.copy()

        for fld in fields:
            setattr(self, fld, np.zeros_like(self.r))