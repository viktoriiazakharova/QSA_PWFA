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
        if (r_axis[l]<=r_axis[j]) and (l!=j):
            sum_result += a[l]

    return sum_result
