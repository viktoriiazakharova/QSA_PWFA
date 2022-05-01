import numpy as np
from tqdm import tqdm

class Simulation:

    def __init__(self, L_xi, N_xi, verbose=1):

        self.verbose = verbose
        self.external_fields = []
        self.species = []
        self.diagnostics = []
        self._init_xi_grid(L_xi, N_xi)

    def add_external_field(self, external_field):
        self.external_fields.append(external_field)

    def add_specie(self, specie):
        self.species.append(specie)

    def run(self, iter_max=30, rel_err_max=1e-2, mixing_factor=0.05):
        for i_xi in tqdm(range(self.N_xi-1)):
            self._advance_xi(iter_max=iter_max, 
                            rel_err_max=rel_err_max,
                            mixing_factor=mixing_factor)

    def _init_xi_grid(self, L_xi, N_xi):
        # iteration counter
        self.i_xi = 0

        # grid range and resolutions
        self.L_xi = L_xi
        self.N_xi = N_xi

        self.xi = L_xi / N_xi * np.arange(N_xi)
        self.dxi = self.xi[1] - self.xi[0]

    def _advance_xi(self, iter_max, rel_err_max, mixing_factor):

        for specie in self.species:
            specie.reinit()
            for specie_src in self.species:
                specie.get_Psi(specie_src)

            specie.get_v_z()

        for specie in self.species:
            for specie_src in self.species:
                specie.get_dPsi_dr(specie_src)
                specie.get_dPsi_dxi(specie_src)
                specie.get_dAz_dr(specie_src)

            for ext_field in self.external_fields:
                specie.dAz_dr += ext_field.get_dAz_dr(specie.r,
                                                      self.xi[self.i_xi])

            specie.get_Fr_part()

        err_rel = 1.0
        i_conv = 0

        while (err_rel>rel_err_max) and (i_conv<iter_max):
            i_conv += 1

            for specie in self.species:
                specie.d2r_dxi2_prev[:] = specie.d2r_dxi2
                specie.dAr_dxi[:] = 0.0

            for specie in self.species:
                for specie_src in self.species:
                    specie.get_dAr_dxi(specie_src)

            for specie in self.species:
                specie.get_Fr()

                specie.get_d2r_dxi2()
                specie.d2r_dxi2 = mixing_factor * specie.d2r_dxi2 + \
                                  (1.0 - mixing_factor) * specie.d2r_dxi2_prev

            err_rel = 0.0
            N_species = len(self.species)
            for specie in self.species:
                err_abs = np.abs(specie.d2r_dxi2 - specie.d2r_dxi2_prev).sum()
                ref_intergal = 0.5 * (np.abs(specie.d2r_dxi2_prev).sum() \
                                    + np.abs(specie.d2r_dxi2).sum() )

                if ref_intergal != 0:
                    err_rel +=  err_abs / ref_intergal / N_species

        if self.verbose>0 and iter_max>0 and (i_conv==iter_max):
            print(f"reached max PC iterations at i_xi={self.i_xi}",
                  f"(xi={self.xi[self.i_xi]}), with an error {err_rel:g}")

        if self.verbose>1 and iter_max>0:
            print(f"reached error {err_rel:g} in {i_conv} PC iterations",
                  f"at i_xi={self.i_xi} (xi={self.xi[self.i_xi]})")

        for specie in self.species:
            specie.advance_motion(self.dxi)

        for diag in self.diagnostics:
            diag.make_record(self.i_xi)

        self.i_xi += 1