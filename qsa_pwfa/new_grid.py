from numba import njit, prange
import numpy as np


@njit
def sum_up_to_r( a, r, r_rings ):
    Nr = r_rings.size
    sum_result = 0

    for l in range(Nr):
        if (r_rings[l]<=r):
            sum_result += a[l]

    return sum_result

@njit
def get_psi_part_ng(r, r_rings, dV):
   
    Psi_r = -0.25 * r**2 + sum_up_to_r( \
            dV * np.log(r / r_rings ), r, r_rings )

    return Psi_r


@njit
def get_psi_ng(r, r_rings, dV, ro):
   
    Psi_r = ( dV * ( ( (r > r0).astype(np.int32)  - (r > r_rings).astype(np.int32) ) * np.log(r0 / r) +  \
                         (r_rings > r).astype(np.int32) * np.log(r_rings / r0)   )).sum()

    return Psi_r


@njit
def get_Psi_new_grid(r, r_rings, dV, r0):
    Psi0 = (dV * np.log(r_rings / r0 )).sum()
    Psi = get_psi_part_ng(r, r_rings, dV)
    Psi += Psi0
    return Psi

@njit    
def get_Psi_new_grid_new_method(r, r_rings, dV, r0):

    Psi = get_psi_ng(r, r_rings, dV, r0)

    return Psi

    