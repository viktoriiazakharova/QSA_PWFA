from numba import njit, prange
import numpy as np

@njit
def fix_crossing_axis_r(r):
    Nr = r.size
    for j in range(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
    return r

@njit
def fix_crossing_axis_rp(r, p):
    Nr = r.size
    for j in range(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            p[j] = np.abs(p[j])
    return r, p

@njit
def sum_up_to_j( a, j, r_axis ):
    Nr = r_axis.size
    sum_result = 0

    for l in range(Nr):
        if (r_axis[l]<=r_axis[j]):
            sum_result += a[l]

    return sum_result

@njit
def sum_up_to_j_2( a, j, r_axis ):
    Nr = r_axis.size
    sum_result = (a * (r_axis<r_axis[j]) ).sum()
    # sum_result = a[(r_axis<r_axis[j])].sum()
    return sum_result

@njit(parallel=True)
def get_psi_part_inline( Psi, r, dV):
    N_r = int(r.size)

    for j in prange(N_r):
        Psi[j] = -0.25 * r[j]**2 + sum_up_to_j( \
            dV * np.log(r[j] / r), j, r )

    return Psi

@njit(parallel=True)
def get_psi_inline( Psi, r, dV):
    N_r = int(r.size)

    for j in prange(N_r):
        Psi[j] = (    ).sum()

    return Psi

@njit(parallel=True)
def get_dAz_dr_part_inline( dAz_dr, r, dV, v_z):
    N_r = int(r.size)

    for j in prange(N_r):
        dAz_dr[j] = sum_up_to_j( dV * v_z / (1-v_z), j, r )/ r[j]

    return dAz_dr

@njit
def slice_density_projection(r, wr, dr, Nr):
    val = np.zeros(Nr)

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

def get_density(r_xi, vz_xi, dV, dr, Nr):

    N_xi, N_rings = r_xi.shape

    r_new = dr*np.arange(Nr)
    dV_new = r_new * dr
    dV_new[0] = dr**2

    dens = np.zeros((N_xi, Nr))

    for i_xi in prange(N_xi):
        dens[i_xi] = slice_density_projection(r_xi[i_xi], dV/(1-vz_xi[i_xi]), dr, Nr)
        dens[i_xi] /= dV_new

    return dens

def get_field(field_val, r_xi, vz_xi, dV, dr, Nr):

    N_xi, N_rings = r_xi.shape

    r_new = dr*np.arange(Nr)
    dV_new = r_new * dr
    dV_new[0] = dr**2

    dens = np.zeros((N_xi, Nr))

    for i_xi in prange(N_xi):
        dens[i_xi] = slice_density_projection(r_xi[i_xi], \
            dV * field_val[i_xi]/(1-vz_xi[i_xi]), dr, Nr)
        dens[i_xi] /= dV_new

    return dens
