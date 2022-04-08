import numpy as np
from .inline_methods import *


class Simulation:

    def __init__(self, L_xi, N_xi, L_r, N_r,
                 dens_func=None, verbose=0):

        self.verbose = verbose
        self.init_grids(L_xi, N_xi, L_r, N_r, dens_func)
        self.allocate_data()
        self.beams = []

    def add_beam(self, beam):
        self.beams.append(beam)

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

        self.dAr_dxi = np.zeros_like(self.r0)
        self.dAz_dr = np.zeros_like(self.r0)
        self.dAz_dr_beams  = np.zeros_like(self.r0)

        self.Psi = np.zeros_like(self.r0)
        self.dPsi_dr = np.zeros_like(self.r0)
        self.dPsi_dxi = np.zeros_like(self.r0)

        self.dr_dxi = np.zeros_like(self.r0)
        self.d2r_dxi2 = np.zeros_like(self.r0)

        self.F = np.zeros_like(self.r0)
        self.F_part = np.zeros_like(self.r0)

    def add_beams_field(self, xi_i, r):
        self.dAz_dr_beams[:] = 0.0
        for beam in self.beams:
            self.dAz_dr_beams += beam.gaussian_integrate(r, xi_i)/ r

        self.dAz_dr += self.dAz_dr_beams

    def get_vz(self, p_perp, Psi):
        self.T[:] = (1. + p_perp ** 2) / (1. + Psi) ** 2
        self.v_z[:] = (self.T - 1.) / (self.T + 1.)

    def get_p_perp(self, dr_dxi, Psi):
        self.p_perp[:] = dr_dxi * (1. + Psi)

    def get_dAz_dr(self, r, v_z):
        self.dAz_dr = get_dAz_dr_inline(self.dAz_dr, r, self.dV, v_z)

    def get_dPsi_dr(self, r):
        self.dPsi_dr = self.get_dPsi_dr_inline(self.dPsi_dr, r, \
                                               self.r0, self.dV)

    def get_Psi(self, r_loc):
        self.Psi = get_psi_inline(self.Psi, r_loc, self.r0, self.dV)

    def get_force_reduced(self):
        self.F_part[:] = self.dPsi_dr + (1. - self.v_z) * self.dAz_dr

    def get_force_full(self):
        self.F[:] = self.F_part + (1. - self.v_z) * self.dAr_dxi
        self.F *= ( self.r<=self.r0.max() )

    def get_d2r_dxi2(self):
        self.d2r_dxi2[:] = ( self.F / (1. - self.v_z) \
            - self.dPsi_dxi * self.dr_dxi ) / (1 + self.Psi)

    def get_dAr_dxi(self, r, dr_dxi, d2r_dxi2):
        self.dAr_dxi = get_dAr_dxi_inline(self.dAr_dxi, r, dr_dxi,
                                          d2r_dxi2, self.dV)

    def get_dPsi_dxi(self, r, dr_dxi):
        self.dPsi_dxi = get_dpsi_dxi_inline(self.dPsi_dxi, r, self.r0,
                                            dr_dxi, self.dV)

    def advance_xi(self, iter_max=30, rel_err_max=1e-2, mixing_factor=0.05):

        # d_Ar/d_xi at i_xi-1 (had it from previous step?)
        self.get_dAr_dxi(self.r, self.dr_dxi, self.d2r_dxi2)

        # advance r and r' at i_xi
        self.r[:] = self.r + self.dxi * self.dr_dxi
        self.dr_dxi[:] = self.dr_dxi + self.d2r_dxi2 * self.dxi

        # fix axis crossing
        fix_crossing_axis_rp( self.r, self.dr_dxi )

        # get Psi, d_Psi/d_r, d_Psi/d_xi, d_Az/d_r, p_r, v_z
        # at i_xi
        self.get_Psi( self.r )
        self.get_p_perp( self.dr_dxi, self.Psi )
        self.get_vz( self.p_perp, self.Psi )

        self.get_dPsi_dr( self.r )
        self.get_dPsi_dxi(self.r, self.dr_dxi)

        self.get_dAz_dr( self.r, self.v_z )
        self.add_beams_field( self.xi[self.i_xi], self.r)

        # get reduced (e.s.) force at i_xi
        self.get_force_reduced()

        # get full force F* -- with d_Ar/d_xi at i_xi-1
        self.get_force_full()

        err_rel = 1.0
        i_conv = 0

        err_rel_list = []
        while (err_rel>rel_err_max) and (i_conv<iter_max):
            i_conv += 1

            Force_prev = self.F.copy()
            d2r_dxi2_tmp = self.d2r_dxi2.copy()

            self.get_d2r_dxi2()
            self.d2r_dxi2 = mixing_factor * self.d2r_dxi2 + \
                                (1.0 - mixing_factor) * d2r_dxi2_tmp
            self.get_dAr_dxi( self.r, self.dr_dxi, self.d2r_dxi2 )
            self.get_force_full()

            err_abs = np.abs(self.F-Force_prev).sum()
            ref_intergal_prev = np.abs(Force_prev).sum()
            ref_intergal_new = np.abs(self.F).sum()

            if ref_intergal_prev==0.0 and ref_intergal_new==0.0:
                err_rel = 0.0
            else:
                err_rel = 2 * err_abs / (ref_intergal_prev + ref_intergal_new)


        if self.verbose>0 and iter_max>0 and (i_conv==iter_max):
            print(f"max iterations is reached with error {err_rel:g}")

        if self.verbose>1 and iter_max>0:
            print(f"reached error {err_rel:g} in {i_conv} iterations")

        self.get_d2r_dxi2()
        self.i_xi += 1