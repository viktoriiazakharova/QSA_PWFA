import numpy as np
from tqdm.auto import tqdm

class Simulation:

    def __init__(self, L_xi=None, N_xi=None, 
                 xi_grid_user=None, verbose=1, dt=0.0):

        self.verbose = verbose
        self.dt = dt
        self.external_fields = []
        self.species = []
        self.diagnostics = []
        self._init_xi_grid(L_xi, N_xi, xi_grid_user)

    def sort_species(self):
        self.species_plasma = []
        self.species_bunch = []
        self.species_grid = []
        for specie in self.species:
            if specie.type == 'Bunch':
                self.species_bunch.append(specie)
            elif specie.type == 'Grid':
                self.species_grid.append(specie)
            else:
                self.species_plasma.append(specie)

    def add_external_field(self, external_field):
        self.external_fields.append(external_field)

    def add_specie(self, specie):
        self.species.append(specie)
        self.sort_species()

    def _init_xi_grid(self, L_xi, N_xi, xi_grid_user):
        if L_xi is not None and  N_xi is not None:
            self.L_xi = L_xi
            self.N_xi = N_xi
            self.xi = L_xi / N_xi * np.arange(N_xi)
        else:
            self.xi = xi_grid_user.copy()
            self.L_xi = self.xi.max()
            self.N_xi = self.xi.size

        self.dxi = np.gradient(self.xi) #self.xi[1] - self.xi[0]

    def run_step(self, iter_max=30, rel_err_max=1e-2, 
                 mixing_factor=0.05, track_convergence=False):

        self.i_xi = 0
        self.track_convergence = track_convergence
        if self.track_convergence:
            self.err_abs_list = []
            self.err_rel_list = []
            self.i_conv_list = []

        for diag in self.diagnostics:
            diag.make_dataset()

        for i_xi in tqdm(range(self.N_xi-1)):
            self._advance_xi(iter_max=iter_max,
                            rel_err_max=rel_err_max,
                            mixing_factor=mixing_factor,
                            do_diag=True)

        for diag in self.diagnostics:
            diag.save_dataset()

        if self.verbose>0:
            for i_specie, specie in enumerate(self.species_plasma):
                if specie.do_QSA_check:
                    fraction_lost = specie.Q_QSA_violate / specie.dQ.sum()
                    print(\
                f"Specie {i_specie} had {fraction_lost*100:g}% violated QSA")

    def run_steps(self, N_steps, iter_max=50, rel_err_max=1e-2,
                  mixing_factor=0.05, track_convergence=False):

        self.track_convergence = track_convergence
        if self.track_convergence:
            self.err_abs_list = []
            self.err_rel_list = []
            self.i_conv_list = []
            
        for specie in self.species_plasma:
            setattr(specie, 'dQ0', specie.dQ.copy())

        with tqdm( total=N_steps*(self.N_xi-1) ) as pbar:

            for i_step in range(N_steps):
                self.i_xi = 0

                for diag in self.diagnostics:
                    if np.mod(i_step, diag.dt_step) == 0:
                        diag.make_dataset()
                        do_diag = True

                for specie in self.species_plasma:
                    specie.refresh_plasma()

                for i_xi in range(self.N_xi-1):
                    self._advance_xi(iter_max=iter_max,
                                    rel_err_max=rel_err_max,
                                    mixing_factor=mixing_factor,
                                    do_diag=do_diag )
                    pbar.update(1)

                for diag in self.diagnostics:
                    if np.mod(i_step, diag.dt_step) == 0:
                        diag.save_dataset()
                        do_diag = False

    def _advance_xi(self, iter_max, rel_err_max, mixing_factor, do_diag):

        for specie in self.species:
            specie.reinit_data(self.i_xi)

        for specie in self.species:
            for specie_src in self.species_plasma:
                specie.get_Psi(specie_src)

        for specie in self.species_plasma:
            specie.get_v_z()
            specie.check_QSA()

        for specie in self.species:
            for specie_src in self.species_plasma:
                specie.get_dPsi_dr(specie_src)
                specie.get_dPsi_dxi(specie_src)

            for specie_src in self.species:
                specie.get_dAz_dr(specie_src)

            for ext_field in self.external_fields:
                specie.dAz_dr += ext_field.get_dAz_dr(specie.r,
                                                      self.xi[self.i_xi])

        for specie in self.species_plasma:
            specie.get_Fr_part()

        err_rel = 1.0
        i_conv = 0
        if self.track_convergence:
            err_abs_list_loc = []
            err_rel_list_loc = []
            i_conv_list_loc = []

        while (err_rel>rel_err_max) and (i_conv<iter_max):
            for specie in self.species:
                specie.dAr_dxi[:] = 0.0

            for specie in self.species_plasma:
                specie.d2r_dxi2_prev[:] = specie.d2r_dxi2

            for specie in self.species_plasma:
                for specie_src in self.species_plasma:
                    specie.get_dAr_dxi(specie_src)

            for specie in self.species_plasma:
                specie.get_Fr()

                specie.get_d2r_dxi2()
                specie.d2r_dxi2 = mixing_factor * specie.d2r_dxi2 + \
                                  (1.0 - mixing_factor) * specie.d2r_dxi2_prev

            err_rel = 0.0
            N_species = len(self.species_plasma)
            for specie in self.species_plasma:
                err_abs = np.abs(specie.d2r_dxi2 - specie.d2r_dxi2_prev).sum()
                ref_intergal = 0.5 * (np.abs(specie.d2r_dxi2_prev).sum() \
                                    + np.abs(specie.d2r_dxi2).sum() )

                if ref_intergal != 0:
                    err_rel +=  err_abs / ref_intergal / N_species

            if self.track_convergence:
                err_abs_list_loc.append(err_abs)
                err_rel_list_loc.append(err_rel)
                i_conv_list_loc.append(i_conv)

            i_conv += 1

        if self.track_convergence:
            self.err_abs_list.append(err_abs_list_loc)
            self.err_rel_list.append(err_rel_list_loc)
            self.i_conv_list.append(i_conv_list_loc)

        if self.verbose>0 and iter_max>0 and (i_conv==iter_max):
            print(f"reached max PC iterations at i_xi={self.i_xi}",
                  f"(xi={self.xi[self.i_xi]}), with an error {err_rel:g}")

        if self.verbose>1 and iter_max>0:
            print(f"reached error {err_rel:g} in {i_conv} PC iterations",
                  f"at i_xi={self.i_xi} (xi={self.xi[self.i_xi]})")

        for specie in self.species_bunch:
            specie.dAr_dxi[:] = 0.0
            for specie_src in self.species_plasma:
                specie.get_dAr_dxi(specie_src)

            specie.advance_motion(self.dt)

        for specie in self.species_plasma:
            specie.advance_motion(self.dxi[self.i_xi])

        if do_diag:
            for diag in self.diagnostics:
                diag.make_record(self.i_xi)

        self.i_xi += 1