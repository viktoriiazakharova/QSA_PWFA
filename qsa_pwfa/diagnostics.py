import numpy as np

class FieldDiagnostics:

    def __init__(self, simulation, grid, fields=['Psi', ],
                 xi_step=1, xi_range=None, species_src=None ):
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

        self.grid = grid
        self.simulation = simulation
        self.fields = fields.copy()

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

        self.Data = {}
        for fld in self.fields:
            self.Data[fld] = np.zeros((self.i_xi.size, grid.N_r))

    def make_record(self, i_xi):

        if i_xi in self.i_xi:
            i_xi_loc = (self.i_xi<i_xi).sum()

            self.grid.init_data(self.fields)
            for fld in self.fields:
                for specie_src in self.species_src:
                    getattr(self.grid, 'get_'+fld)(specie_src)

                self.Data[fld][i_xi_loc] = getattr(self.grid, fld)
        else:
            return
