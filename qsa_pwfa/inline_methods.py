from numba import njit, prange
import numpy as np

@njit
def fast_less(x, x1):
    val = np.zeros(x.size, dtype=np.float32)
    for ix in range(val.size):
        if x[ix] < x1:
            val[ix] = 1.0
        elif x[ix] == x1:
            val[ix] = 0.5
        else:
            val[ix] = 0.0
    return val

@njit
def fast_less_int(x, x1):
    val = np.zeros(x.size, dtype=np.int8)
    for ix in range(val.size):
        if x[ix] <= x1:
            val[ix] = 1
        else:
            val[ix] = 0
    return val

@njit(parallel=True)
def fix_crossing_axis_rp(r, p):
    Nr = r.size
    for j in prange(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            p[j] = np.abs(p[j])
    return r, p

@njit(parallel=True)
def get_psi_inline( Psi_target, r_target, r_source, r0_source, dV_source ):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        H_r_m_r0j = fast_less(r0_source, r_target[ir])

        Psi_target[ir] += np.sum( dV_source * \
            ( ( H_r_m_r0j - H_r_m_rj ) * np.log(r0_source / r_target[ir]) \
              + H_rj_m_r_excl * np.log(r_source / r0_source) ) )

    return Psi_target

@njit(parallel=True)
def get_dPsi_dr_unif_inline(dPsi_dr_target, r_target, r_source,
                            r0_source, dV_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])

        dPsi_dr_target[ir] += -0.5  * r_target[ir] + \
            np.sum(dV_source * H_r_m_rj ) / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dPsi_dr_inline(dPsi_dr_target, r_target, r_source,
                       r0_source, dV_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])
        H_r_m_r0j = fast_less(r0_source, r_target[ir])        
        dPsi_dr_target[ir] += np.sum(dV_source * ( H_r_m_rj - H_r_m_r0j ) )\
                             / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dAz_dr_inline(dAz_dr_target, r_target, r_source,
                      v_z_source, dV_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])
        dAz_dr_target[ir] += np.sum(dV_source * v_z_source / (1.-v_z_source) \
                                * H_r_m_rj ) / r_target[ir]

    return dAz_dr_target

@njit(parallel=True)
def get_dPsi_dxi_inline(dPsi_dxi_target, r_target, r_source,
                        r0_source, dr_dxi_source, dV_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        dPsi_dxi_target[ir] += np.sum( dV_source * dr_dxi_source \
                                      * H_rj_m_r_excl / r_source)

    return dPsi_dxi_target

@njit(parallel=True)
def get_dAr_dxi_inline(dAr_dxi_target, r_target, r_source, dr_dxi_source,
                       d2r_dxi2_source, dV_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_less(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj        
        r_source_inv = 1./r_source
        dAr_dxi_target[ir] = -1. / r_target[ir] * np.sum ( dV_source \
            * ( dr_dxi_source**2 * H_r_m_rj \
              + 0.5 * ( d2r_dxi2_source * r_source_inv \
                       - (dr_dxi_source * r_source_inv) ** 2 ) \
              * (r_target[ir]**2 * H_rj_m_r_excl  + r_source**2 * H_r_m_rj)
            ) )

    return dAr_dxi_target