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
    
@njit(parallel=True)
def get_psi_part_inline( Psi, r, dV):
    N_r = int(r.size)
    
    for j in prange(N_r):
        Psi[j] = -0.25 * r[j]**2 + sum_up_to_j( \
            dV * np.log(r[j] / r), j, r )
        
    return Psi

@njit(parallel=True)
def get_dAz_dr_part_inline( dAz_dr, r, dV, v_z):
    N_r = int(r.size)
    
    for j in prange(N_r):
        dAz_dr[j] = sum_up_to_j( dV * v_z / (1-v_z), j, r )/ r[j]
        
    return dAz_dr
        
       
        
        