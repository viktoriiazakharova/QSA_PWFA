from numba import njit, prange
import numpy as np

@njit(parallel=True)
def fix_crossing_axis_r(r):
    Nr = r.size
    for j in prange(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
    return r

@njit(parallel=True)
def fix_crossing_axis_rp(r, p):
    Nr = r.size
    for j in prange(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            p[j] = np.abs(p[j])
    return r, p


@njit(parallel=True)
def get_dPsi_dr_inline(dPsi_dr, r, dV):
    Nr = r.size
    for ir in prange(Nr):
        dPsi_dr[ir] = -0.5  * r[ir] + \
            (dV * (r <= r[ir]) ).sum() / r[ir]

    return dPsi_dr


@njit(parallel=True)
def get_dAz_dr_part_inline(dAz_dr, r, dV, v_z):
    Nr = r.size
    for ir in prange(Nr):
        dAz_dr[ir] = (dV * v_z / (1-v_z) * (r <= r[ir]) ).sum() / r[ir]

    return dAz_dr

@njit(parallel=True)
def get_psi_inline( Psi, r, r0, dV):
    N_r = int(r.size)

    for j in prange(N_r):
        Psi[j] = ( dV * \
            ( ( (r[j] > r0).astype(np.int8) - (r[j] > r).astype(np.int8) ) * \
              np.log(r0 / r[j]) +  \
            (r > r[j]).astype(np.int8) * np.log(r / r0)  )).sum()

    return Psi