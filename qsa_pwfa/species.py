from ast import Return
import numpy as np
from .inline_methods import fix_crossing_axis_rv
from .inline_methods import fix_crossing_axis_rvp
from .inline_methods import fix_crossing_axis_rvpxy
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


class BunchSliceRZ(BaseSpecie):

    def __init__(self, r, xi, p_r, p_z, dQ):
        
        self.type = "Bunch"
        self.fields = BaseSpecie.base_fields

        self.r = r.copy()
        self.r0 = self.r
        self.xi = xi.copy()
        
        self.p_r = p_r.copy()
        self.p_z = p_z.copy()
        self.dQ = dQ.copy()

        gamma_inv = 1.0 / np.sqrt( 1.0 + p_r*p_r + p_z*p_z )

        self.v_z = p_z * gamma_inv
        self.dr_dxi = p_r * gamma_inv

        if r.size>0:
            self.rmax = r.max()
        else:
            self.rmax = 0.0

        self.d2r_dxi2 = 0.0

    def advance_motion(self, dt):

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


class BunchSpecieRZ_PPC():

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

        r_bunch = r[None,:] * np.ones_like(xi)[:, None]
        xi_bunch = xi[:, None] * np.ones_like(r)[None, :]

        dQ_bunch = dr0 * (r_bunch - 0.5 * dr0)
        dQ_bunch[:, 0] = 0.125 * dr0[0]**2

        dQ_bunch *= self.q * self.n_p * self.dens_func(\
            r_bunch - 0.5 * dr0, xi_bunch)

        p_r_bunch = self.eps_r / self.sigma_r \
            * np.random.randn(*r_bunch.shape)
        gamma_p = self.gamma_b + self.delta_gamma \
            * np.random.randn(*r_bunch.shape)
        p_z_bunch = (gamma_p**2 - 1. - p_r_bunch**2)**0.5

        self.empty_slice = BunchSliceRZ( np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0) )
        self.empty_slice.n_cycles = self.n_cycles
        self.empty_slice.n_p = self.n_p
        self.empty_slice.q = self.q
        self.empty_slice.particle_boundary = self.particle_boundary

        self.bunch_slices = {}
        for i_xi in range(r_bunch.shape[0]):
            self.bunch_slices[i_xi] = BunchSliceRZ( r_bunch[i_xi],
                                                    xi_bunch[i_xi],
                                                    p_r_bunch[i_xi],
                                                    p_z_bunch[i_xi],
                                                    dQ_bunch[i_xi] )

            self.bunch_slices[i_xi].n_cycles = self.n_cycles
            self.bunch_slices[i_xi].n_p = self.n_p
            self.bunch_slices[i_xi].q = self.q
            self.bunch_slices[i_xi].particle_boundary = self.particle_boundary

    def reinit_data(self, i_xi):
        if (i_xi >= self.i_xi_min) and (i_xi <= self.i_xi_max):
            self.local_slice = self.bunch_slices[i_xi - self.i_xi_min]
        else:
            self.local_slice = self.empty_slice

        self.local_slice.init_data(self.local_slice.fields)

class GaussianBunchRZ_PPC(BunchSpecieRZ_PPC):
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


class BunchSlice3D(BaseSpecie):

    def __init__(self, x, y, xi, p_x, p_y, p_z, dQ):
        
        self.type = "Bunch"
        self.fields = BaseSpecie.base_fields

        self.x = x.copy()
        self.y = y.copy()
        self.xi = xi.copy()
        
        self.p_x = p_x.copy()
        self.p_y = p_y.copy()
        self.p_z = p_z.copy()
        self.dQ = dQ.copy()

        self.r = np.sqrt( x*x + y*y )
        self.p_r = ( p_x*x + p_y*y ) / self.r

        self.r0 = self.r

        gamma_inv = 1.0 / np.sqrt( 1.0 + p_x*p_x + p_y*p_y + p_z*p_z )

        self.v_z = p_z * gamma_inv
        self.dr_dxi = self.p_r * gamma_inv

        if self.r.size>0:
            self.rmax = self.r.max()
        else:
            self.rmax = 0.0

        self.d2r_dxi2 = 0.0

    def advance_motion(self, dt):
        Ez = self.dPsi_dxi
        Er = -self.dPsi_dr - self.dAz_dr - self.dAr_dxi
        Bt = -self.dAz_dr - self.dAr_dxi

        Ex =  Er * self.x/self.r
        Ey =  Er * self.y/self.r
        Bx = -Bt * self.y/self.r
        By =  Bt * self.x/self.r

        #self.Ez = Ez
        #self.Ex = Ex
        #self.Ey = Ey
        #self.Bx = Ex
        #self.By = Ey

        self.p_x += 0.5 * self.q * dt * Ex
        self.p_y += 0.5 * self.q * dt * Ey
        self.p_z += 0.5 * self.q * dt * Ez

        gamma_inv = 1.0 / np.sqrt(1. + self.p_x**2 + self.p_y**2 + self.p_z**2)

        vx_mid = self.p_x * gamma_inv
        vy_mid = self.p_y * gamma_inv
        vz_mid = self.p_z * gamma_inv

        self.p_x += self.q * dt * ( 0.5*Ex - vz_mid*By )
        self.p_y += self.q * dt * ( 0.5*Ey + vz_mid*Bx )
        self.p_z += self.q * dt * ( 0.5*Ez + vx_mid*By - vy_mid*Bx )

        gamma_inv = 1.0 / np.sqrt(1. + self.p_x**2 + self.p_y**2 + self.p_z**2)

        self.v_z[:] = self.p_z * gamma_inv

        self.x += self.p_x * gamma_inv * dt
        self.y += self.p_y * gamma_inv * dt
        self.xi += ( self.v_z - 1 ) * dt

        self.r[:] = np.sqrt(self.x*self.x + self.y*self.y)
        self.p_r[:] = ( self.p_x * self.x + self.p_y * self.y ) / self.r
        self.dr_dxi[:] = self.p_r * gamma_inv
        fix_crossing_axis_rvpxy( self.r, self.dr_dxi, self.p_r,
                                 self.x, self.y, self.p_x, self.p_y )

class BunchFromArrays(BaseSpecie):
    """
    User class to create Gaussian bunch species.
    """

    fields = BaseSpecie.base_fields

    def __init__( self, simulation,
                  x, y, xi, p_x, p_y, p_z, charge_total,
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

        self.Np = xi.size
        self.xi_min, self.xi_max = xi.min(), xi.max()
        self.i_xi_min = (self.simulation.xi <= self.xi_min).sum()
        self.i_xi_max = (self.simulation.xi <= self.xi_max).sum()

        dQ = q * charge_total * np.ones_like(x) / self.Np / (2*np.pi)

        indx_sort = np.argsort(xi)
        xi_sim = self.simulation.xi[(self.simulation.xi <= self.xi_max) \
                                  * (self.simulation.xi >= self.xi_min)]
        dxi_sim = self.simulation.dxi[(self.simulation.xi <= self.xi_max) \
                                  * (self.simulation.xi >= self.xi_min)]

        indicies_in_slices = {}
        for i_xi_sim in range(len(xi_sim)):
            indicies_in_slices[i_xi_sim] = []

        for i_p in indx_sort:
            xi_p = xi[i_p]
            i_xi_loc = np.abs(xi_p - xi_sim).argmin()
            indicies_in_slices[i_xi_loc].append(i_p)

        for i_xi_sim in range(len(xi_sim)):
            indicies_in_slice = indicies_in_slices[i_xi_sim]
            dQ[indicies_in_slice] /= dxi_sim[i_xi_sim]

        self.bunch_slices = {}
        for i_xi in range(len(xi_sim)):
            indicies_in_slice = indicies_in_slices[i_xi]
            self.bunch_slices[i_xi] = BunchSlice3D( x  [indicies_in_slice], 
                                                    y  [indicies_in_slice], 
                                                    xi [indicies_in_slice],
                                                    p_x[indicies_in_slice],
                                                    p_y[indicies_in_slice],
                                                    p_z[indicies_in_slice],
                                                    dQ [indicies_in_slice])

            self.bunch_slices[i_xi].n_cycles = self.n_cycles
            self.bunch_slices[i_xi].n_p = self.n_p
            self.bunch_slices[i_xi].q = self.q
            self.bunch_slices[i_xi].particle_boundary = self.particle_boundary

        self.empty_slice = BunchSlice3D( np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0),
                                         np.zeros(0) )
        self.empty_slice.n_cycles = self.n_cycles
        self.empty_slice.n_p = self.n_p
        self.empty_slice.q = self.q
        self.empty_slice.particle_boundary = self.particle_boundary

    def reinit_data(self, i_xi):
        if (i_xi >= self.i_xi_min) and (i_xi < self.i_xi_max):
            self.local_slice = self.bunch_slices[i_xi - self.i_xi_min]
        else:
            self.local_slice = self.empty_slice

        self.local_slice.init_data(self.local_slice.fields)