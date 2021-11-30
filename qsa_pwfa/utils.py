from numba import njit, prange
import numpy as np

@njit(parallel=True)
def get_Psi_new_grid(Psi, r_grid, r, r0, dV):
    Nr = r.size

    for ir in prange(Nr):
        r_loc = r_grid[ir]
        Psi[ir] = ( dV * \
          ( \
            ( (r_loc >= r0).astype(np.int8) - (r_loc >= r).astype(np.int8) ) * \
              np.log( r0 / r_loc ) +  \
            (r > r_loc).astype(np.int8) * np.log( r / r0 ) ) \
          ).sum()

    return Psi

@njit
def slice_density_projection(val, r, wr, dr, Nr):
    N_rings = int(r.size)

    for i_ring in range(N_rings):
        r_i = r[i_ring]
        ir_cell = int(np.floor(r_i/dr))
        if ir_cell>Nr-2:
            continue

        s1 = r_i / dr - ir_cell
        s0 = 1. - s1
        val[ir_cell] += wr[i_ring] * s0
        val[ir_cell+1] += wr[i_ring] * s1

    return val

@njit(parallel=True)
def get_density(r_xi, vz_xi, dV, dr, Nr):

    N_xi, N_rings = r_xi.shape
    dens = np.zeros((N_xi, Nr))

    r_new = dr*np.arange(Nr)
    dV_new = r_new * dr
    dV_new[0] = dr**2

    for i_xi in prange(N_xi):
        dens_loc = np.zeros(Nr)
        dens_loc = slice_density_projection(dens_loc, r_xi[i_xi], \
            dV/(1-vz_xi[i_xi]), dr, Nr)
        dens[i_xi,:] = dens_loc / dV_new

    return dens

@njit(parallel=True)
def get_field(field_val, r_xi, vz_xi, dV, dr, Nr):

    N_xi, N_rings = r_xi.shape

    r_new = dr*np.arange(Nr)
    dV_new = r_new * dr
    dV_new[0] = dr**2

    dens = np.zeros((N_xi, Nr))

    for i_xi in prange(N_xi):
        dens_loc = np.zeros(Nr)
        dens_loc = slice_density_projection(dens_loc, r_xi[i_xi], \
            dV * field_val[i_xi]/(1-vz_xi[i_xi]), dr, Nr)
        dens[i_xi,:] = dens_loc/dV_new

    return dens
