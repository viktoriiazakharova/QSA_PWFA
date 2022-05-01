import numpy as np
from .inline_methods import *


class BaseSpecie:

    base_fields = [
        'Psi',
        'dPsi_dr',
        'dPsi_dxi',
        'dAr_dxi',
        'dAz_dr',
        'Fr',
        'Fr_part',
        ]
            
    def init_data(self, fields):
        for fld in fields:
            setattr(self, fld, np.zeros_like(self.r))

    def init_r_grid(self, L_r, N_r, r_grid_user):

        if (L_r is not None) and (N_r is not None):
            self.L_r = L_r
            self.N_r = N_r
            self.r0 = L_r / N_r * np.arange(1, N_r+1)
            self.dr0 = np.gradient(self.r0)

            self.r0 -= 0.5*self.dr0
            self.r = self.r0.copy()
            self.dQ = self.dr0 * (self.r0 - 0.5*self.dr0)
            self.dQ[0] = 0.125 * self.dr0[0]**2

            self.rmax = self.r0.max()

        elif r_grid_user is not None:
            self.r0 = r_grid_user.copy()
            self.r = self.r0.copy()
            self.rmax = self.r0.max()
            self.L_r = self.rmax
            self.N_r = self.r0.size
            self.dr0 = np.gradient(self.r0)
            self.dQ = self.dr0 * (self.r0 - 0.5*self.dr0)
            self.dQ[0] = 0.5 * self.dr0[0]**2
        else:
            print('need to define the grid')

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
        else:
            return                                    

    def get_dPsi_dxi(self, source_specie):
        self.dPsi_dxi = get_dPsi_dxi_inline(self.dPsi_dxi, self.r,
                                            source_specie.r,
                                            source_specie.r0,
                                            source_specie.dr_dxi,
                                            source_specie.dQ)

    def get_dAr_dxi(self, source_specie):
        self.dAr_dxi = get_dAr_dxi_inline(self.dAr_dxi, self.r,
                                          source_specie.r,
                                          source_specie.dr_dxi,
                                          source_specie.d2r_dxi2,
                                          source_specie.dQ)

    def get_Fr_part(self):
        self.Fr_part[:] = self.dPsi_dr + (1. - self.v_z) * self.dAz_dr

    def get_Fr(self):
        self.Fr[:] = self.Fr_part + (1. - self.v_z) * self.dAr_dxi
        if self.particle_boundary == 1:
            self.Fr *= ( self.r<=self.rmax )


class PlasmaSpecie(BaseSpecie):

    motion_fields = [
        'v_z',
        'dr_dxi',
        'd2r_dxi2',
        'd2r_dxi2_prev',
        ]

    fields = motion_fields + BaseSpecie.base_fields

    def reinit(self):
        self.init_data(self.base_fields)

    def get_v_z(self):
        T = (1. + (self.dr_dxi * (1. + self.Psi)) ** 2) / \
            (1. + self.Psi) ** 2
        self.v_z[:] = (T - 1.) / (T + 1.)

    def get_d2r_dxi2(self):
        self.d2r_dxi2[:] = ( self.Fr / (1. - self.v_z) \
            - self.dPsi_dxi * self.dr_dxi ) / (1 + self.Psi)

    def advance_motion(self, dxi):
            self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
            self.r += self.dr_dxi * dxi
            self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
            fix_crossing_axis_rp(self.r, self.dr_dxi)


class NeutralUniformPlasma(PlasmaSpecie):

    def __init__(self, L_r=None, N_r=None, r_grid_user=None, n_p=1,
                 particle_boundary=1):

        self.type = "NeutralUniformPlasma"

        self.particle_boundary = particle_boundary
        self.n_p = n_p

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= n_p

        self.init_data(self.fields)


class NeutralNoneUniformPlasma(PlasmaSpecie):

    def __init__(self, dens_func, L_r=None, N_r=None, r_grid_user=None,
                 particle_boundary=0):

        self.type = "NeutralNoneUniformPlasma"

        self.particle_boundary = particle_boundary
        self.dens_func = dens_func

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= dens_func(self.r0)

        self.init_data(self.fields)

        
class Grid(BaseSpecie):

    def __init__(self, L_r=None, N_r=None, r_grid_user=None):

        self.particle_boundary = 0
        self.type = "Grid"
        self.init_r_grid(L_r, N_r, r_grid_user)

    def get_Density(self, source_specie):
        weights = source_specie.dQ / (1 - source_specie.v_z)
        Density_loc = np.zeros_like(self.Density)
        Density_loc = get_Density_inline(Density_loc, self.r0, self.dr0,
                                         source_specie.r, weights)
        Density_loc /= self.dQ
        self.Density += Density_loc

    def get_J_z(self, source_specie):
        weights = source_specie.dQ * source_specie.v_z/ (1 - source_specie.v_z)
        J_z_loc =  np.zeros_like(self.J_z)
        J_z_loc = get_Density_inline(J_z_loc, self.r0, self.dr0,
                                     source_specie.r, weights)
        J_z_loc /= self.dQ
        self.J_z += J_z_loc

    def get_v_z(self, source_specie):
        weights = source_specie.dQ / (1 - source_specie.v_z)
        weights_vz = weights * source_specie.v_z
        dens_temp = np.zeros_like(self.r0)
        dens_temp = get_Density_inline(dens_temp, self.r0, self.dr0,
                                      source_specie.r, weights)
        self.v_z = get_Density_inline(self.v_z, self.r0, self.dr0,
                                      source_specie.r, weights_vz)
        self.v_z /= dens_temp