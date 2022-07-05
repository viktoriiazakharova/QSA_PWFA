from ast import Return
import numpy as np
from .inline_methods import fix_crossing_axis_rv
from .inline_methods import fix_crossing_axis_rvp
from .inline_methods import methods_inline


class BaseSpecie:
    """
    Main and base class for the particle objects.

    Contains following methods:
      init_data: creates attributes associated with the fields
        as empty r-like arrays.
      init_r_grid: generates the radial grid.
      get_[field term] : wrapper methods that call numba-compiled functions that
        calculate various [field terms] for electromagnetic field.
    """

    base_fields = [
        'Psi',
        'dPsi_dr',
        'dPsi_dxi',
        'dAr_dxi',
        'dAz_dr',
        ]

    def init_data(self, fields):
        for fld in fields:
            setattr(self, fld, np.zeros_like(self.r))

    def init_r_grid(self, L_r, N_r, r_grid_user):

        if r_grid_user is not None:
            self.user_grid = True
        elif (L_r is not None) and (N_r is not None):
            self.user_grid = False
        else:
            print('missing parameters to define the grid')
            return

        if self.user_grid:
            self.r0 = r_grid_user.copy()
            if (self.r0[0] == 0.0):
                self.r0 +=  0.5 * self.r0[1]
                print('warning: origin of r_grid_user is adjusted')            
            
            self.dr0 = np.zeros_like(self.r0)
            self.dr0[1:] = self.r0[1:] - self.r0[:-1]
            self.dr0[0] = self.r0[0]
            self.N_r = self.r0.size
            self.L_r = self.r0.max()
        else:
            self.L_r = L_r
            self.N_r = N_r
            self.r0 = L_r / N_r * np.arange(N_r)
            self.dr0 = L_r / N_r * np.ones_like(self.r0)
            self.r0 += 0.5 * self.dr0

        self.dV = self.dr0 * (self.r0 - 0.5*self.dr0)
        self.dV[0] = 0.125 * self.dr0[0]**2      
        self.dQ = self.dV.copy()
        
        self.r = self.r0.copy()
        self.rmax = self.r.max()

    def get_dAz_dr(self, source_specie):
        self.dAz_dr = methods_inline[source_specie.type]['dAz_dr'](
                self.dAz_dr, self.r,
                source_specie.r,
                source_specie.v_z,
                source_specie.dQ)

    def get_Psi(self, source_specie):
        self.Psi = methods_inline[source_specie.type]['Psi'](
                                self.Psi, self.r,
                                source_specie.r,
                                source_specie.r0,
                                source_specie.dQ)

    def get_dPsi_dr(self, source_specie):
        self.dPsi_dr = methods_inline[source_specie.type]['dPsi_dr'](
                                self.dPsi_dr, self.r,
                                source_specie.n_p,
                                source_specie.r,
                                source_specie.r0,
                                source_specie.dQ)

    def get_dPsi_dxi(self, source_specie):
        self.dPsi_dxi = methods_inline[source_specie.type]['dPsi_dxi'](
                                self.dPsi_dxi, self.r,
                                source_specie.r,
                                source_specie.r0,
                                source_specie.dr_dxi,
                                source_specie.dQ)

    def get_dAr_dxi(self, source_specie):
        self.dAr_dxi = methods_inline[source_specie.type]['dAr_dxi'](
                                self.dAr_dxi, self.r,
                                source_specie.r,
                                source_specie.dr_dxi,
                                source_specie.d2r_dxi2,
                                source_specie.dQ)


class PlasmaSpecie(BaseSpecie):
    """
    Generic class for plasma QSA particle species.
    
    Contains following methods:
      reinit_data: reinitialize the electromagnetic fields.
      get_v_z: calculate `v_z` of QSA particles.
      check_QSA: check and suppress the particles that violate QSA.
      get_Fr_part: calculate the force on QSA particles without 
        account for the implicit term`dAr_dxi`.
      get_Fr: calculate the full force on QSA particles.
      get_d2r_dxi2: calculate the radial acceleration of QSA particles.
      advance_motion: advance radial coordinates and velocities of 
        QSA particles and treat the axis crossing.
      refresh_plasma: reset all plasma attributes to initial state.
    """

    motion_fields = [
        'v_z',
        'Fr',
        'Fr_part',
        'dr_dxi',
        'd2r_dxi2',
        'd2r_dxi2_prev',
        ]

    fields = motion_fields + BaseSpecie.base_fields

    def reinit_data(self, i_xi=None):
        self.init_data(self.base_fields)

    def get_v_z(self):
        T = (1. + (self.dr_dxi * (1. - self.q * self.Psi)) ** 2) / \
            (1. - self.q * self.Psi) ** 2
        self.v_z[:] = (T - 1.) / (T + 1.)

    def check_QSA(self):
        if self.do_QSA_check:
            self.Q_QSA_violate += self.dQ[self.v_z>self.vz_max_QSA].sum()
            self.dQ[self.v_z>self.vz_max_QSA] = 0.0
            self.v_z[self.v_z>self.vz_max_QSA] = self.vz_max_QSA

    def get_Fr_part(self):
        self.Fr_part[:] = -self.q * (self.dPsi_dr + (1. - self.v_z) * self.dAz_dr)

    def get_Fr(self):
        self.Fr[:] = self.Fr_part - self.q *  (1. - self.v_z) * self.dAr_dxi
        if self.particle_boundary == 1:
            self.Fr *= ( self.r<=self.rmax )

    def get_d2r_dxi2(self):
        self.d2r_dxi2[:] = ( self.Fr / (1. - self.v_z) \
            + self.q *  self.dPsi_dxi * self.dr_dxi ) / (1 - self.q *  self.Psi)

    def advance_motion(self, dxi):
        self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
        self.r += self.dr_dxi * dxi
        self.dr_dxi += 0.5 * self.d2r_dxi2 * dxi
        fix_crossing_axis_rv(self.r, self.dr_dxi)

    def refresh_plasma(self):
        self.r[:] = self.r0
        self.dQ[:] = self.dQ0
        self.init_data(self.fields)

class BunchSpecie(BaseSpecie):
    """
    Generic class for relativistic bunch particle species.

    Contains following methods:
      init_particles: initialize all bunch particles in
        the (xi, r) space and create the slice attributes.
      reinit_data: load the slice at a given xi position.
      advance_motion: advance coordinates and velocities of
        bunch particles in time and treat the axis crossing.
      get_Delta: calculate normalized variation of the
        longitudinal electric field over the slice and weighted
        with particle charges.
    """

    fields = BaseSpecie.base_fields

    def init_particles(self):

        self.xi_min = self.xi_0 - self.truncate_factor * self.sigma_xi
        self.xi_max = self.xi_0 + self.truncate_factor * self.sigma_xi
        self.i_xi_min = (self.simulation.xi <= self.xi_min).sum()
        self.i_xi_max = (self.simulation.xi <= self.xi_max).sum()

        xi = self.simulation.xi[self.i_xi_min : self.i_xi_max + 1]
        L_r = self.truncate_factor * self.sigma_r

        r = L_r / self.N_r * np.arange(self.N_r)
        dr0 = L_r / self.N_r * np.ones_like(r)
        r += 0.5 * dr0

        self.r_bunch = r[None,:] * np.ones_like(xi)[:, None]
        self.xi_bunch = xi[:, None] * np.ones_like(r)[None, :]

        self.dQ_bunch = dr0 * (self.r_bunch - 0.5 * dr0)
        self.dQ_bunch[:, 0] = 0.125 * dr0[0]**2

        self.dV = self.dQ_bunch[0, :].copy()

        self.dQ_bunch *= self.q * self.n_p * self.dens_func(\
            self.r_bunch - 0.5 * dr0, self.xi_bunch)

        self.p_r_bunch = self.eps_r / self.sigma_r \
            * np.random.randn(*self.r_bunch.shape)
        gamma_p = self.gamma_b + self.delta_gamma \
            * np.random.randn(*self.r_bunch.shape)
        self.p_z_bunch = (gamma_p**2 - 1. - self.p_r_bunch**2)**0.5

        self.v_z_bunch = self.p_z_bunch / gamma_p
        self.dr_dxi_bunch = self.p_r_bunch / gamma_p

        self.rmax = r.max()
        self.r = r.copy()
        self.r0 = r.copy()
        self.v_z = np.ones_like(self.r)
        self.dQ = np.zeros_like(self.r)
        self.p_z  = np.zeros_like(self.r)
        self.p_r  = np.zeros_like(self.r)
        self.dr_dxi = np.zeros_like(self.r)
        self.d2r_dxi2 = 0.0

    def reinit_data(self, i_xi):
        self.r = np.zeros(0)
        self.xi = np.zeros(0)
        self.p_r  = np.zeros(0)
        self.p_z = np.zeros(0)
        self.v_z = np.zeros(0)
        self.dQ = np.zeros(0)
        self.dr_dxi = np.zeros(0)

        if (i_xi >= self.i_xi_min) and (i_xi <= self.i_xi_max):
            self.r = self.r_bunch[i_xi - self.i_xi_min]
            self.xi = self.xi_bunch[i_xi - self.i_xi_min]
            self.dQ = self.dQ_bunch[i_xi - self.i_xi_min]
            self.v_z = self.v_z_bunch[i_xi - self.i_xi_min]
            self.p_z = self.p_z_bunch[i_xi - self.i_xi_min]
            self.p_r = self.p_r_bunch[i_xi - self.i_xi_min]
            self.dr_dxi = self.dr_dxi_bunch[i_xi - self.i_xi_min]

        self.init_data(self.fields)

    def advance_motion(self, dt):
        # Ez = self.dPsi_dxi
        # Er = -self.dPsi_dr - self.dAz_dr - self.dAr_dxi
        # Bt = -self.dAz_dr - self.dAr_dxi

        self.Fz = self.q * dt * (self.dPsi_dxi \
            - self.dr_dxi * (self.dAz_dr - self.dAr_dxi) )
        self.Fr = - self.q * dt \
            * (self.dPsi_dr + (self.dAz_dr + self.dAr_dxi) * (1 - self.v_z) )

        self.p_z += 0.5 * self.Fz
        self.p_r += 0.5 * self.Fr

        gamma_p = np.sqrt(1. + self.p_z**2 + self.p_r**2)
        self.v_z[:] = self.p_z / gamma_p
        self.dr_dxi[:] = self.p_r  / gamma_p

        self.xi += (self.v_z-1) * dt
        self.r += self.dr_dxi * dt

        self.p_z += 0.5 * self.Fz
        self.p_r += 0.5 * self.Fr

        gamma_p = np.sqrt(1. + self.p_z**2 + self.p_r**2)
        self.v_z[:] = self.p_z / gamma_p
        self.dr_dxi[:] = self.p_r  / gamma_p

        fix_crossing_axis_rvp(self.r, self.dr_dxi, self.p_r)

    def get_Delta(self):
        _Ez_ = np.average(self.dPsi_dxi, weights=self.dQ)
        _Ez2_ = np.average((self.dPsi_dxi - _Ez_)**2, weights=self.dQ)
        Delta = _Ez2_**0.5 / _Ez_

        return Delta


class NeutralUniformPlasma(PlasmaSpecie):
    """
    User class to create the neutral uniform plasma species.
    """

    def __init__(self, L_r=None, N_r=None, r_grid_user=None, n_p=1.0,
                 particle_boundary=1, q=-1.0, max_weight_QSA=35.0):
        """
        Initialize the plasma particles.

        Args:
            L_r (float, optional): Radial size of the plasma. Is only used
              for uniform grids. Defaults to None, which assumes that 
              `r_grid_user` is used instead.
            N_r (integer, optional): Number of particles that presents the
              size of the radial grid. Is only used for uniform grids. 
              Defaults to None, which assumes that `r_grid_user` is used 
              instead.
            r_grid_user (float ndarray, optional): Radial grid that presents 
              the plasma particles. Defaults to None, which assumes that 
              `L_r` and `N_r` are used instead.
            n_p (float, optional): Plasma density in the units of reference
              plasma density. Defaults to 1.0.
            particle_boundary (int, optional): If set to 1, the particles that
              cross initial plasma radius do not experience the radial force
              (still carrying the charge). Defaults to 1.
            q (float, optional): Charge of plasma moving species in units of
              elementary charge `e`. Defaults to -1.0.
            max_weight_QSA (float, optional): Set the limit for the particle
              _weigth_ allowed by QSA, and defined as `gamma / (Psi + 1)`.
              Defaults to 35.0.
        """

        self.type = "NeutralUniformPlasma"
        self.n_p = n_p
        self.q = q
        self.particle_boundary = particle_boundary

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= n_p
        self.dQ *= self.q

        self.init_data(self.fields)

        if max_weight_QSA is not None:
            self.do_QSA_check = True
            self.vz_max_QSA = 1. - 1./max_weight_QSA
            self.Q_QSA_violate = 0
        else:
            self.do_QSA_check = False

class NeutralNoneUniformPlasma(PlasmaSpecie):
    """
    User class to create the neutral non-uniform plasma species.
    """

    def __init__(self, dens_func, L_r=None, N_r=None, r_grid_user=None,
                 particle_boundary=0, q=-1.0, max_weight_QSA=35.0):
        """
        Initialize the plasma particles.

        Args:
            dens_func (function): Function of radial coordinate that defines
              plasma density profile
            L_r (float, optional): Radial size of the plasma. Is only used
              for uniform grids. Defaults to None, which assumes that 
              `r_grid_user` is used instead.
            N_r (integer, optional): Number of particles that presents the
              size of the radial grid. Is only used for uniform grids. 
              Defaults to None, which assumes that `r_grid_user` is used 
              instead.
            r_grid_user (float ndarray, optional): Radial grid that presents 
              the plasma particles. Defaults to None, which assumes that 
              `L_r` and `N_r` are used instead.
            particle_boundary (int, optional): If set to 1, the particles that
              cross initial plasma radius do not experience the radial force
              (still carrying the charge). Defaults to 1.
            q (float, optional): Charge of plasma moving species in units of
              elementary charge `e`. Defaults to -1.0.
            max_weight_QSA (float, optional): Set the limit for the particle 
              _weigth_ allowed by QSA, and defined as `gamma / (Psi + 1)`.
              Defaults to 35.0.
        """

        self.type = "NeutralNoneUniformPlasma"
        self.particle_boundary = particle_boundary
        self.q = q

        self.init_r_grid(L_r, N_r, r_grid_user)
        self.dQ *= dens_func(self.r0)
        self.dQ *= self.q
        self.n_p = dens_func(self.r0).max()

        self.init_data(self.fields)

        if max_weight_QSA is not None:
            self.do_QSA_check = True
            self.vz_max_QSA = 1. - 1./max_weight_QSA
            self.Q_QSA_violate = 0
        else:
            self.do_QSA_check = False

class GaussianBunch(BunchSpecie):
    """
    User class to create Gaussian bunch species.
    """
    def __init__( self, simulation, n_p, sigma_r, sigma_xi,
                  xi_0=None, N_r=512, gamma_b=1e4, q=-1.0, 
                  delta_gamma=0.0, eps_r=0.0, n_cycles=1,
                  truncate_factor=4.0 ):
        """
        Initialize the Gaussian bunch.

        Args:
            simulation (Simulation): Simulation object
            n_p (float): Maximum charge density in units of reference 
              plasma density.
            sigma_r (float): radial RSM size of the bunch.
            sigma_xi (float): longitudinal RSM size of the bunch
            xi_0 (float, optional): Initial position of the bunch. If None,
              the beam is set as driver at `xi = truncate_factor * sigma_xi`.
              Defaults to None.
            N_r (int, optional): Number of particles along radial axis.
              Defaults to 512.
            gamma_b (float, optional): Mean Lorentz factor of the bunch.
              Defaults to 10 000.
            q (float, optional): Charge of bunch particle species in units of
              elementary charge `e`. Defaults to -1.0.
            delta_gamma (float, optional): Relative RMS spread of Lorentz factor
              in the bunch. Defaults to 0.0.
            eps_r (float, optional): Normalized emittance of the bunch.
              Defaults to 0.0.
            n_cycles (int, optional): Number of sub-steps to be performed for
              bunch motion over the time step. Defaults to 1.
            truncate_factor (float, optional): Factor that defines how far from
              the bunch center the particles are created. In units of `sigma_xi` and
              `sigma_r`. Defaults to 4.0.
        """

        self.type = "Bunch"
        self.n_cycles = n_cycles
        self.particle_boundary = 0
        self.simulation = simulation
        self.n_p = n_p
        self.q = q
        self.N_r = N_r
        self.sigma_r = sigma_r
        self.sigma_xi = sigma_xi

        if xi_0 is not None:
            self.xi_0 = xi_0
        else:
            self.xi_0 = truncate_factor * sigma_xi

        self.gamma_b = gamma_b
        self.delta_gamma = delta_gamma
        self.eps_r = eps_r
        self.truncate_factor = truncate_factor
        self.dens_func = self.dens_func_gauss
        self.init_particles()

    def dens_func_gauss(self, r_bunch, xi_bunch):
        val = np.exp( - 0.5 * r_bunch**2 / self.sigma_r**2 \
                      - 0.5 * (xi_bunch - self.xi_0)**2 / self.sigma_xi**2 )
        return val


class Grid(BaseSpecie):
    """
    Specific class of a Grid used by FieldDiagnostics.
    """

    def __init__(self, L_r=None, N_r=None, r_grid_user=None):
        """_summary_

        Args:
            L_r (float, optional): Radial size of the grid. Is only used
              for uniform grids. Defaults to None, which assumes that 
              `r_grid_user` is used instead.
            N_r (integer, optional): Size of the radial grid. 
              Is only used for uniform grids. Defaults to None, which assumes
              that `r_grid_user` is used instead.
            r_grid_user (float ndarray, optional): Radial grid. Defaults 
              to None, which assumes that `L_r` and `N_r` are used instead.
        """

        self.particle_boundary = 0
        self.type = "Grid"
        self.init_r_grid(L_r, N_r, r_grid_user)

    def get_Density(self, source_specie):
        if source_specie.type == 'Bunch':
            weights = source_specie.dQ
        else:
            weights = source_specie.dQ / (1 - source_specie.v_z)
        Density_loc = np.zeros_like(self.Density)

        Density_loc = methods_inline[source_specie.type]['Density'](
                                    Density_loc, self.r0, self.dr0,
                                    source_specie.r, weights)
        Density_loc /= self.dV
        self.Density += Density_loc

    def get_J_z(self, source_specie):
        if source_specie.type == 'Bunch':
            weights = source_specie.dQ * source_specie.v_z
        else:
            weights = source_specie.dQ * source_specie.v_z \
                    / (1 - source_specie.v_z)
        J_z_loc =  np.zeros_like(self.J_z)
        J_z_loc = methods_inline[source_specie.type]['Density'](
                                    J_z_loc, self.r0, self.dr0,
                                    source_specie.r, weights)
        J_z_loc /= self.dV
        self.J_z += J_z_loc

    def get_v_z(self, source_specie):
        if source_specie.type == 'Bunch':
            weights = source_specie.dQ
        else:
            weights = source_specie.dQ / (1 - source_specie.v_z)

        weights_vz = weights * source_specie.v_z
        dens_temp = np.zeros_like(self.r0)
        dens_temp = methods_inline[source_specie.type]['Density'](
                                    dens_temp, self.r0, self.dr0,
                                    source_specie.r, weights)

        self.v_z = methods_inline[source_specie.type]['Density'](
                                    self.v_z, self.r0, self.dr0,
                                    source_specie.r, weights_vz)
        self.v_z /= dens_temp


class BunchFromArrays(BaseSpecie):  
    """
    User class to create Gaussian bunch species.
    """
    
    fields = BaseSpecie.base_fields

    def __init__( self, simulation, 
                  x, y, xi, ux, uy, uz, charge_total, 
                  q=-1.0, n_cycles=1 ):
        """
        Initialize the Gaussian bunch.

        Args:
            simulation (Simulation): Simulation object
            n_cycles (int, optional): Number of sub-steps to be performed for
              bunch motion over the time step. Defaults to 1.
        """

        self.type = "Bunch"
        self.n_cycles = n_cycles
        self.particle_boundary = 0
        self.simulation = simulation
        self.n_p = 1.0
        self.q = q

        # self.xi_0 = np.average(z, weights=wght)
        # self.xi_bunch = 2 * self.xi_0 - z

        self.Np = xi.size
        self.xi_bunch = xi
        self.xi_min, self.xi_max = self.xi_bunch.min(), self.xi_bunch.max()
        self.i_xi_min = (self.simulation.xi <= self.xi_min).sum()
        self.i_xi_max = (self.simulation.xi <= self.xi_max).sum()

        self.x_bunch = x
        self.y_bunch = y
        self.ux_bunch = ux
        self.uy_bunch = uy

        self.r_bunch = np.sqrt( x*x + y*y )
        self.p_r_bunch = (ux*x + uy*y) / self.r_bunch

        self.p_z_bunch = uz
        gamma_bunch = np.sqrt(1.0 + ux*ux + uy*uy + uz*uz)

        self.v_z_bunch = self.p_z_bunch / gamma_bunch
        self.dr_dxi_bunch = self.p_r_bunch / gamma_bunch

        self.dQ_bunch = q * charge_total * np.ones_like(x) / self.Np / (2*np.pi)
        self.init_particles()

    def init_particles(self):
        indx_sort = np.argsort(self.xi_bunch)
        xi_sim = self.simulation.xi[(self.simulation.xi <= self.xi_max) \
                                  * (self.simulation.xi >= self.xi_min)]
        dxi_sim = self.simulation.dxi[(self.simulation.xi <= self.xi_max) \
                                  * (self.simulation.xi >= self.xi_min)]

        self.indicies_in_slices = {}
        for i_xi_sim in range(len(xi_sim)):
            self.indicies_in_slices[i_xi_sim] = []

        for i_p in indx_sort:
            xi_p = self.xi_bunch[i_p]
            i_xi_loc = np.abs(xi_p - xi_sim).argmin()
            self.indicies_in_slices[i_xi_loc].append(i_p)

        for i_xi_sim in range(len(xi_sim)):
            indicies_in_slice = self.indicies_in_slices[i_xi_sim]
            self.dQ_bunch[indicies_in_slice] /= dxi_sim[i_xi_sim]

        self.rmax = self.r_bunch.max()
        self.r = self.r_bunch[self.indicies_in_slices[0]].copy()
        self.r0 = self.r.copy()
        self.v_z = np.ones_like(self.r)
        self.dQ = np.zeros_like(self.r)
        self.p_z = np.zeros_like(self.r)
        self.p_r = np.zeros_like(self.r)
        self.dr_dxi = np.zeros_like(self.r)
        self.d2r_dxi2 = 0.0

    def reinit_data(self, i_xi):
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.xi = np.zeros(0)
        self.ux = np.zeros(0)
        self.uy = np.zeros(0)
        self.p_z = np.zeros(0)
        self.v_z = np.zeros(0)
        self.dQ = np.zeros(0)
        self.r = np.zeros(0)
        self.p_r = np.zeros(0)
        self.dr_dxi = np.zeros(0)

        if (i_xi >= self.i_xi_min) and (i_xi < self.i_xi_max):
            indicies_in_slice = self.indicies_in_slices[i_xi - self.i_xi_min]
            if len(indicies_in_slice)>0:
                self.x = self.x_bunch[indicies_in_slice]
                self.y = self.y_bunch[indicies_in_slice]
                self.xi = self.xi_bunch[indicies_in_slice]
                self.ux = self.ux_bunch[indicies_in_slice]
                self.uy = self.uy_bunch[indicies_in_slice]
                self.p_z = self.p_z_bunch[indicies_in_slice]
                self.v_z = self.v_z_bunch[indicies_in_slice]
                self.dQ = self.dQ_bunch[indicies_in_slice]
                self.r = self.r_bunch[indicies_in_slice]
                self.p_r = self.p_r_bunch[indicies_in_slice]
                self.dr_dxi = self.dr_dxi_bunch[indicies_in_slice]

        self.init_data(self.fields)

    def advance_motion_new(self, dt):
        Ez = self.dPsi_dxi
        Er = -self.dPsi_dr - self.dAz_dr - self.dAr_dxi
        Bt = -self.dAz_dr - self.dAr_dxi

        Ex =  Er * self.x/self.r
        Ey =  Er * self.y/self.r
        Bx = -Bt * self.y/self.r
        By =  Bt * self.x/self.r

        self.ux  += 0.5 * self.q * dt * Ex
        self.uy  += 0.5 * self.q * dt * Ey
        self.p_z += 0.5 * self.q * dt * Ez

        gamma_inv = 1.0 / np.sqrt(1. + self.ux**2 + self.uy**2 + self.p_z**2)

        vx_mid = self.ux * gamma_inv
        vy_mid = self.uy * gamma_inv
        vz_mid = self.p_z * gamma_inv

        self.ux  += self.q * dt * ( 0.5*Ex - vz_mid*By )
        self.uy  += self.q * dt * ( 0.5*Ey + vz_mid*Bx )
        self.p_z += self.q * dt * ( 0.5*Ez + vx_mid*By - vy_mid*Bx )

        gamma_inv = 1.0 / np.sqrt(1. + self.ux**2 + self.uy**2 + self.p_z**2)

        self.x += self.ux * gamma_inv * dt
        self.y += self.ux * gamma_inv * dt
        self.xi += ( self.p_z*gamma_inv -1 ) * dt

        self.r = np.sqrt(self.x*self.x + self.y*self.y)
        self.p_r[:] = ( self.ux * self.x + self.uy * self.y ) / self.r
        self.dr_dxi[:] = self.p_r * gamma_inv

        self.v_z[:] = self.p_z * gamma_inv
        self.dr_dxi[:] = self.p_r * gamma_inv

        fix_crossing_axis_rvp(self.r, self.dr_dxi, self.p_r)
        
    def advance_motion(self, dt):
        # Ez = self.dPsi_dxi
        # Er = -self.dPsi_dr - self.dAz_dr - self.dAr_dxi
        # Bt = -self.dAz_dr - self.dAr_dxi

        self.Fz = self.q * dt * (self.dPsi_dxi \
            - self.dr_dxi * (self.dAz_dr - self.dAr_dxi) )
        self.Fr = - self.q * dt \
            * (self.dPsi_dr + (self.dAz_dr + self.dAr_dxi) * (1 - self.v_z) )

        self.p_z += 0.5 * self.Fz
        self.p_r += 0.5 * self.Fr

        gamma_p = np.sqrt(1. + self.p_z**2 + self.p_r**2)
        self.v_z[:] = self.p_z / gamma_p
        self.dr_dxi[:] = self.p_r  / gamma_p

        self.xi += (self.v_z-1) * dt
        self.r += self.dr_dxi * dt

        self.p_z += 0.5 * self.Fz
        self.p_r += 0.5 * self.Fr

        gamma_p = np.sqrt(1. + self.p_z**2 + self.p_r**2)
        self.v_z[:] = self.p_z / gamma_p
        self.dr_dxi[:] = self.p_r  / gamma_p

        fix_crossing_axis_rvp(self.r, self.dr_dxi, self.p_r)