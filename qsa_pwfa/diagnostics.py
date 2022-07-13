import numpy as np
from .species import Grid

from copy import deepcopy

class FieldDiagnostics:

    def __init__( self, simulation, fields=['Psi', ],
                  L_r=None, N_r=None, r_grid_user=None,
                  xi_step=1, xi_range=None, dt_step=1,
                  species_src=None ):
        """
        Available fields are:
          'Density'
          'J_z'
          'v_z'
          'Psi'
          'dPsi_dxi'
          'dPsi_dr'
          'dAr_xi'
          'dAz_dr'

        NB: 'v_z' works only with a single specie, so it has
        to be defined in a `species_src` list.
        """

        self.grid = Grid(L_r, N_r, r_grid_user)
        self.simulation = simulation
        self.dt_step = dt_step
        self.do_diag = True
        self.fields = fields.copy()
        self.outputs = []

        if species_src is not None:
            self.species_src = species_src
        else:
            self.species_src = simulation.species

        if xi_range is not None:
            xi_select = (simulation.xi>=xi_range[0]) \
                      * (simulation.xi<=xi_range[1])
            self.i_xi = np.arange(simulation.xi.size)[xi_select][::xi_step]
        else:
            self.i_xi = np.arange(simulation.xi.size)[::xi_step]

        self.xi = simulation.xi[self.i_xi]
        self.grid.init_data(self.fields)

    def make_dataset(self):
        self.Data = {}
        for fld in self.fields:
            self.Data[fld] = np.zeros((self.xi.size, self.grid.N_r))

    def save_dataset(self):
        self.outputs.append(deepcopy(self.Data))

    def make_record(self, i_xi):

        if i_xi in self.i_xi:
            i_xi_loc = np.nonzero(self.i_xi == i_xi)[0]

            self.grid.init_data(self.fields)
            for fld in self.fields:
                for specie_src in self.species_src:
                    if specie_src.type=='Bunch':
                        getattr(self.grid, 'get_'+fld)(specie_src.local_slice)
                    else:
                        getattr(self.grid, 'get_'+fld)(specie_src)

                self.Data[fld][i_xi_loc] = getattr(self.grid, fld)
        else:
            return


class SpeciesDiagnostics:

    def __init__( self, simulation, specie, fields=['Psi', ],
                  xi_step=1, xi_range=None, species_src=None, dt_step=1 ):
        """
        Available fields are:
          'v_z'
          'Psi'
          'dPsi_dxi'
          'dPsi_dr'
          'dAr_xi'
          'dAz_dr'
        """

        self.specie = specie
        self.grid = specie
        self.simulation = simulation
        self.dt_step = dt_step
        self.fields = fields.copy()
        self.outputs = []
        self.do_diag = True

        if species_src is not None:
            self.species_src = species_src
        else:
            self.species_src = simulation.species

        if xi_range is not None:
            xi_select = (simulation.xi>=xi_range[0]) \
                      * (simulation.xi<=xi_range[1])
            self.i_xi = np.arange(simulation.xi.size)[xi_select][::xi_step]
        else:
            self.i_xi = np.arange(simulation.xi.size)[::xi_step]

        self.xi = simulation.xi[self.i_xi]

    def make_dataset(self):
        self.Data = {}
        self.Data['r'] = np.zeros((self.i_xi.size, self.specie.r0.size))
        self.Data['xi'] = self.xi.copy()
        self.Data['dQ'] = self.specie.dQ.copy()
        self.Data['r0'] = self.specie.r0.copy()
        for fld in self.fields:
            self.Data[fld] = np.zeros((self.i_xi.size, self.specie.r0.size))

    def save_dataset(self):
        self.outputs.append(deepcopy(self.Data))

    def make_record(self, i_xi):
        i_xi_loc = np.nonzero(self.i_xi == i_xi)[0]
        if i_xi_loc.size>0:
            i_xi_loc = i_xi_loc[0]
            N_r = self.specie.r.size
            self.Data['r'][i_xi_loc, :N_r] = self.specie.r.copy()
            for fld in self.fields:
                self.Data[fld][i_xi_loc, :N_r] = getattr(self.specie, fld)
        else:
            return


class BunchDiagnostics:

    def __init__( self, simulation, bunch,
                  fields=['xi', 'r', 'p_z', 'p_r'],
                  species_src=None, dt_step=1 ):
        """
        Available fields are:
          'v_z'
          'Psi'
          'dPsi_dxi'
          'dPsi_dr'
          'dAr_xi'
          'dAz_dr'
          'Delta'
        """

        self.bunch = bunch
        self.simulation = simulation
        self.dt_step = dt_step
        self.do_diag = True
        self.fields = fields.copy()
        self.outputs = []

        if species_src is not None:
            self.species_src = species_src
        else:
            self.species_src = simulation.species

        self.i_xi = np.arange(simulation.xi.size)
        self.i_xi = self.i_xi[(self.i_xi>=self.bunch.i_xi_min)*(self.i_xi<=self.bunch.i_xi_max)]
        self.xi = simulation.xi[self.i_xi]

    def make_dataset(self):
        self.Data = {}

        for fld in self.fields + ['dQ', ]:
            self.Data[fld] = np.zeros(0)

    def save_dataset(self):
        self.outputs.append(deepcopy(self.Data))

    def make_record(self, i_xi):

        i_xi_loc = np.nonzero(self.i_xi == i_xi)[0]
        if i_xi_loc.size>0:
            Np_loc = self.Data['dQ'].size
            Np_new = self.bunch.local_slice.r.size

            for fld in self.fields + ['dQ', ]:
                self.Data[fld].resize(Np_loc+Np_new, refcheck=False)
                self.Data[fld][Np_loc:] = getattr(self.bunch.local_slice, fld)


class BunchParametersDiagnostics:

    def __init__( self, simulation, bunch,
                  fields=['sigma_x', 'sigma_y', 'epsilon_x', 'epsilon_y'],
                  species_src=None, dt_step=1 ):
        """
        """

        self.bunch = bunch
        self.simulation = simulation
        self.dt_step = dt_step
        self.do_diag = True
        self.fields = fields.copy()  + ['sliceQ', ]
        self.outputs = []

        if species_src is not None:
            self.species_src = species_src
        else:
            self.species_src = simulation.species

        self.i_xi = np.arange(simulation.xi.size)
        self.i_xi = self.i_xi[(self.i_xi>=self.bunch.i_xi_min)*(self.i_xi<=self.bunch.i_xi_max)]
        self.xi = simulation.xi[self.i_xi]

    def make_dataset(self):
        self.Data = {}

        for fld in self.fields:
            self.Data[fld] = []

    def save_dataset(self):
        self.outputs.append(deepcopy(self.Data))

    def make_record(self, i_xi):
        i_xi_loc = np.nonzero(self.i_xi == i_xi)[0]
        if i_xi_loc.size>0:
            for fld in self.fields:
                if self.bunch.local_slice.dQ.size>1:
                    self.Data[fld].append( getattr(self.bunch.local_slice,
                                                   'get_'+fld)() )
                else:
                    self.Data[fld].append( 0.0 )