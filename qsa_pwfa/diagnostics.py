import numpy as np

class FieldDiagnostics:

    def __init__(self, simulation, grid, fields=['Psi', ], xi_step=1 ):
        self.grid = grid
        self.simulation = simulation
        self.fields = fields.copy()
        self.xi_step = xi_step
        self.N_xi = int(simulation.N_xi / xi_step)

        self.Data = {}
        for fld in self.fields:
            self.Data[fld] = np.zeros((self.N_xi, grid.N_r))

    def make_record(self, xi):
        if np.mod(xi, self.xi_step)!=0:
            return

        xi_loc = xi//self.xi_step
        self.grid.reinit()
        for fld in self.fields:
            for specie_src in self.simulation.species:
                getattr(self.grid, 'get_'+fld)(specie_src)

            self.Data[fld][xi_loc] = getattr(self.grid, fld)
