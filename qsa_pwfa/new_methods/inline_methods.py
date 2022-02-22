from numba import njit, prange
import numpy as np

@njit(parallel=True)
def fix_crossing_axis_rp(r, p):
    Nr = r.size
    for j in prange(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            p[j] = np.abs(p[j])
    return r, p

@njit(parallel=True)
def get_dPsi_dr_unif_inline(dPsi_dr, r, r0, dV):
    Nr = r.size
    for ir in prange(Nr):
        dPsi_dr[ir] = -0.5  * r[ir] + \
            np.sum(dV * (r <= r[ir]) ) / r[ir]

    return dPsi_dr

@njit(parallel=True)
def get_dPsi_dr_inline(dPsi_dr, r, r0, dV):
    Nr = r.size
    for ir in prange(Nr):
        dPsi_dr[ir] = np.sum(dV * ( (r <= r[ir]).astype(np.int8) \
            - (r0 <= r[ir]).astype(np.int8) ) ) / r[ir]

    return dPsi_dr

@njit(parallel=True)
def get_dAz_dr_inline(dAz_dr, r, dV, v_z):
    Nr = r.size
    for ir in prange(Nr):
        dAz_dr[ir] = np.sum(dV * v_z / (1.-v_z) * (r <= r[ir]) ) / r[ir]

    return dAz_dr

@njit(parallel=True)
def get_psi_inline( Psi, r, r0, dV):
    N_r = int(r.size)

    for j in prange(N_r):
        Psi[j] = np.sum( dV * \
            ( ( (r0 <= r[j]).astype(np.int8) - (r <= r[j]).astype(np.int8) ) * \
              np.log(r0 / r[j]) +  \
            (r > r[j]).astype(np.int8) * np.log(r / r0)  ))

    return Psi


@njit(parallel=True)
def get_dpsi_dxi_inline( dpsi_dxi, r, r0, dr_dxi, dV):
    N_r = int(r.size)

    for j in prange(N_r):
        dpsi_dxi[j] = np.sum( dV * dr_dxi * (r > r[j]) / r)

    return dpsi_dxi


@njit(parallel=True)
def get_dAr_dxi_inline(dAr_dxi, r, dr_dxi, d2r_dxi2, dV):
    Nr = r.size
    for ir in prange(Nr):
        dAr_dxi[ir] = 1/r[ir] * np.sum ( dV * \
            ( dr_dxi**2 * (r <= r[ir]) \
              + 0.5 * (d2r_dxi2 / r - (dr_dxi / r) ** 2) \
                * ( r[ir]**2 * (r >= r[ir]) + r**2 * (r <= r[ir]) )
            ) )

    return dAr_dxi
