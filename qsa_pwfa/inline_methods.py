from numba import njit, prange
import numpy as np

@njit
def fast_compare(x, x1):
    val = np.zeros(x.size, dtype=np.float32)
    for ix in range(val.size):
        if x[ix] < x1:
            val[ix] = 1.0
        elif x[ix] == x1:
            val[ix] = 0.5
        else:
            val[ix] = 0.0
    return val


@njit(parallel=True)
def fix_crossing_axis_rp(r, p):
    Nr = r.size
    for j in prange(Nr):
        if r[j] < 0:
            r[j] = np.abs(r[j])
            p[j] = np.abs(p[j])
    return r, p

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

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        H_r_m_r0j = fast_compare(r0_source, r_target[ir])

        Psi_target[ir] += np.sum( dQ_source * \
            ( ( H_r_m_r0j - H_r_m_rj ) * np.log(r0_source / r_target[ir]) \
              + H_rj_m_r_excl * np.log(r_source / r0_source) ) )

    return Psi_target

@njit(parallel=True)
def get_dPsi_dr_unif_inline(dPsi_dr_target, r_target, n_p_source, r_source,
                            r0_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])

        dPsi_dr_target[ir] += -0.5  * n_p_source * r_target[ir] + \
            np.sum(dQ_source * H_r_m_rj ) / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dPsi_dr_inline(dPsi_dr_target, r_target, n_p_source, r_source,
                       r0_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_r_m_r0j = fast_compare(r0_source, r_target[ir])
        dPsi_dr_target[ir] += np.sum(dQ_source * ( H_r_m_rj - H_r_m_r0j ) )\
                             / r_target[ir]

    return dPsi_dr_target

@njit(parallel=True)
def get_dAz_dr_inline(dAz_dr_target, r_target, r_source,
                      v_z_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        dAz_dr_target[ir] += np.sum(dQ_source * v_z_source / (1.-v_z_source) \
                                * H_r_m_rj ) / r_target[ir]

    return dAz_dr_target

@njit(parallel=True)
def get_dAz_dr_bunch_inline(dAz_dr_target, r_target, r_source,
                      v_z_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        dAz_dr_target[ir] += np.sum(dQ_source * v_z_source \
                                * H_r_m_rj ) / r_target[ir]

    return dAz_dr_target

@njit(parallel=True)
def get_dPsi_dxi_inline(dPsi_dxi_target, r_target, r_source,
                        r0_source, dr_dxi_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        dPsi_dxi_target[ir] += np.sum( dQ_source * dr_dxi_source \
                                      * H_rj_m_r_excl / r_source)

    return dPsi_dxi_target

@njit(parallel=True)
def get_dAr_dxi_inline(dAr_dxi_target, r_target, r_source, dr_dxi_source,
                       d2r_dxi2_source, dQ_source):

    for ir in prange(r_target.size):
        H_r_m_rj =  fast_compare(r_source, r_target[ir])
        H_rj_m_r_excl = 1 - H_r_m_rj
        r_source_inv = 1./r_source
        dAr_dxi_target[ir] += -1. / r_target[ir] * np.sum ( dQ_source \
            * ( dr_dxi_source**2 * H_r_m_rj \
              + 0.5 * ( d2r_dxi2_source * r_source_inv \
                       - (dr_dxi_source * r_source_inv) ** 2 ) \
              * (r_target[ir]**2 * H_rj_m_r_excl  + r_source**2 * H_r_m_rj)
            ) )

    return dAr_dxi_target

def dummy_function(val, *args, **kw_args):
    return val

methods_inline = {
    "Density":{
        "Grid": dummy_function,  # dummy
        "Bunch": get_Density_inline, # dummy
        "NeutralUniformPlasma": get_Density_inline,
        "NeutralNoneUniformPlasma": get_Density_inline,
    },
    "Psi":{
        "Grid": dummy_function,  # dummy
        "Bunch": dummy_function, # dummy
        "NeutralUniformPlasma": get_Psi_inline,
        "NeutralNoneUniformPlasma": get_Psi_inline,
    },
    "dPsi_dr":{
        "Grid": dummy_function, # dummy
        "Bunch": dummy_function,     # dummy
        "NeutralUniformPlasma": get_dPsi_dr_unif_inline,
        "NeutralNoneUniformPlasma": get_dPsi_dr_inline,
    },
    "dAz_dr":{
        "Grid": dummy_function, # dummy
        "Bunch": get_dAz_dr_bunch_inline, 
        "NeutralUniformPlasma": get_dAz_dr_inline,
        "NeutralNoneUniformPlasma": get_dAz_dr_inline,
    },
    "dPsi_dxi":{
        "Grid": dummy_function, # dummy
        "Bunch": dummy_function,  # dummy
        "NeutralUniformPlasma": get_dPsi_dxi_inline,
        "NeutralNoneUniformPlasma": get_dPsi_dxi_inline,
    },
    "dAr_dxi":{
        "Grid": dummy_function, # dummy
        "Bunch": dummy_function,  # dummy
        "NeutralUniformPlasma": get_dAr_dxi_inline,
        "NeutralNoneUniformPlasma": get_dAr_dxi_inline,
    },    
}