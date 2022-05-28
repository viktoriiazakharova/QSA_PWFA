from numba import njit, prange
import numpy as np

@njit
def fast_compare(x, x1):
    val = np.zeros(x.size, dtype=np.int8)
    for ix in range(val.size):
        if x[ix] <= x1:
            val[ix] = 1
        else:
            val[ix] = 0
    return val

@njit(parallel=True)
def fix_crossing_axis_rv(r, vr):
    for j in prange(r.size):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            vr[j] = np.abs(vr[j])
    return r, vr

@njit(parallel=True)
def fix_crossing_axis_rvp(r, vr, pr):
    for j in prange(r.size):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            vr[j] = np.abs(vr[j])
            pr[j] = np.abs(pr[j])
    return r, vr, pr

@njit
def get_Density_inline(Density_target, r_target, dr_target,
                       r_source, dW_source):

    for ir in range(r_source.size):
        r = r_source[ir]
        ir_cell = (r_target<=r).sum() - 1

        if ir_cell == -1:
            Density_target[0] += dW_source[ir]
        elif ir_cell == r_target.size-1:
            Density_target[ir_cell] += dW_source[ir]
        elif ir_cell>r_target.size-1:
            continue
        else:
            dr = dr_target[ir_cell]
            s1 = (r - r_target[ir_cell]) / dr
            s0 = 1. - s1
            Density_target[ir_cell] += dW_source[ir] * s0
            Density_target[ir_cell+1] += dW_source[ir] * s1

    return Density_target

@njit(parallel=True)
def get_Psi_inline( Psi_target, r_target, r_source, r0_source, dQ_source ):
    r0_source_inv = 1. / r0_source
    log_r_r0_source = np.log(r_source * r0_source_inv)

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        H_r_m_r0j = fast_compare(r0_source, r_target[ir])

        Psi_target[ir] += np.sum( dQ_source * \
            ( ( H_r_m_r0j - H_r_m_rj ) * np.log(r_target[ir] * r0_source_inv) \
              - H_rj_m_r_excl * log_r_r0_source ) )

    return Psi_target

@njit(parallel=True)
def get_dPsi_dr_unif_inline(dPsi_dr_target, r_target, n_p_source, r_source,
                            r0_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])

        dPsi_dr_target[ir] += -0.5  * n_p_source * r_target[ir] - \
            np.sum(dQ_source * H_r_m_rj ) / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dPsi_dr_inline(dPsi_dr_target, r_target, n_p_source, r_source,
                       r0_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_r_m_r0j = fast_compare(r0_source, r_target[ir])
        dPsi_dr_target[ir] += np.sum(dQ_source * ( H_r_m_r0j - H_r_m_rj ) )\
                             / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dAz_dr_inline(dAz_dr_target, r_target, r_source,
                      v_z_source, dQ_source):
    
    dJz_source = dQ_source * v_z_source / (1.-v_z_source)

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        dAz_dr_target[ir] += - np.sum(dJz_source * H_r_m_rj ) / r_target[ir]

    return dAz_dr_target

@njit(parallel=True)
def get_dAz_dr_bunch_inline(dAz_dr_target, r_target, r_source,
                      v_z_source, dQ_source):
    
    v_z_dQ_source = dQ_source * v_z_source

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        dAz_dr_target[ir] += -np.sum(v_z_dQ_source * H_r_m_rj ) / r_target[ir]

    return dAz_dr_target

@njit(parallel=True)
def get_dPsi_dxi_inline(dPsi_dxi_target, r_target, r_source,
                        r0_source, dr_dxi_source, dQ_source):
    
    dQ_dr_dxi_r_source = dQ_source * dr_dxi_source/r_source

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        dPsi_dxi_target[ir] += -np.sum( dQ_dr_dxi_r_source \
                                      * H_rj_m_r_excl)

    return dPsi_dxi_target

@njit(parallel=True)
def get_dAr_dxi_inline(dAr_dxi_target, r_target, r_source, dr_dxi_source,
                       d2r_dxi2_source, dQ_source):
    
    
    r_source_inv = 1./r_source
    dr_dxi_source2 = dr_dxi_source**2
    r_source2 = r_source**2
    
    Term2 = 0.5 * ( d2r_dxi2_source * r_source_inv \
                - (dr_dxi_source * r_source_inv) ** 2 )
    
    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        
        dAr_dxi_target[ir] += 1. / r_target[ir] * np.sum ( dQ_source \
            * ( dr_dxi_source2 * H_r_m_rj \
              + Term2 * (r_target[ir]**2 * H_rj_m_r_excl  + r_source2 * H_r_m_rj)
            ) )

    return dAr_dxi_target

def dummy_function(val, *args, **kw_args):
    return val

methods_inline = {
     "NeutralUniformPlasma": {
         "Density": get_Density_inline,
         "Psi": get_Psi_inline,
         "dPsi_dr": get_dPsi_dr_unif_inline,
         "dAz_dr": get_dAz_dr_inline,
         "dPsi_dxi": get_dPsi_dxi_inline,
         "dAr_dxi": get_dAr_dxi_inline,
     },
     "NeutralNoneUniformPlasma": {
         "Density": get_Density_inline,
         "Psi": get_Psi_inline,
         "dPsi_dr": get_dPsi_dr_inline,
         "dAz_dr": get_dAz_dr_inline,
         "dPsi_dxi": get_dPsi_dxi_inline,
         "dAr_dxi": get_dAr_dxi_inline,
     },
     "Bunch": {
         "Density": get_Density_inline,
         "Psi": dummy_function,
         "dPsi_dr": dummy_function,
         "dAz_dr": get_dAz_dr_bunch_inline,
         "dPsi_dxi": dummy_function,
         "dAr_dxi": dummy_function,
     },
     "Grid": {
         "Density": dummy_function,
         "Psi": dummy_function,
         "dPsi_dr": dummy_function,
         "dAz_dr": dummy_function,
         "dPsi_dxi": dummy_function,
         "dAr_dxi": dummy_function,
     },
}