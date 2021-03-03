import numpy as np
from numba import njit, prange
import matplotlib

class QSA_PWFA:
    def __init__(self, L_xi, N_xi, L_r, N_r):

        self.init_grids(L_xi, N_xi, L_r, N_r)

        self.rainbow = matplotlib.cm.get_cmap('gist_rainbow', self.N_r)

    def init_bunch(self, n_b, R_b, ksi0, R_xi):
        self.n_b = n_b
        self.R_b = R_b
        self.ksi0 = ksi0
        self.R_xi = R_xi
        
    def init_grids(self, L_xi, N_xi, L_r, N_r):
        self.L_xi = L_xi
        self.N_xi = N_xi
        self.L_r = L_r
        self.N_r = N_r
        
        self.xi = L_xi / N_xi * np.arange(N_xi)
        
        self.r0 = np.linspace(L_r/N_r, L_r, N_r)
        
        self.r = self.r0.copy()
        self.r_next = self.r0.copy()
        
        self.dr_dxi_0 = np.zeros_like(self.r0)
        self.dr_dxi_1 = np.zeros_like(self.r0)
        
        self.integral_nb_dr = np.zeros_like(self.r0)
        self.integral_plasma = np.zeros_like(self.r0)
        
        self.crossed0_flag = np.zeros_like(self.r0).astype(np.bool)
        
        # self.r2_grid = self.r0_grid.copy()
        
        self.dxi = self.xi[1] - self.xi[0]    
        self.dr0 = self.r0[1] - self.r0[0]
        
        self.r_xi = np.zeros((N_xi, N_r))  
        
    def gaussian_beam(self, r, ksi):
        """
        Gaussian beam density distribution
        """
        val = self.n_b * np.exp(-(ksi-self.ksi0)**2 / 2 / self.R_xi**2 )\
           * np.exp(-r**2 / (2 * self.R_b**2))

        return val

    def gaussian_integrate(self, r, ksi):
        """
        Gaussian beam density distribution integrated over `r`
        """    
        val = self.n_b * np.exp(-(ksi-self.ksi0)**2 / 2 / self.R_xi**2)\
        * self.R_b**2 * ( 1 - np.exp(-r**2 / 2 / self.R_b**2 ) )
        return val

    def get_dAz_dr(self):
        for j in range(self.N_r):
            self.integral_nb_dr[j] = self.gaussian_integrate(self.r[j], xi_i)
    
    def xi_step(self, i):
            
            xi_i = self.xi[i]


  
                integral_plasma = 0.0
                
                #for l in range(self.N_r): 
                #    if (self.r[l]<=self.r[j]) and (l!=j):
                #        integral_plasma += self.r0[l] * self.dr0  
                        
                integral_plasma = sum_up_to_j( self.r0 * self.dr0 , j, self.r )

                self.dr_dxi_1[j] = self.dr_dxi_0[j] - \
                    0.5 * self.dxi * self.r[j] + self.dxi / self.r[j] * \
                     ( integral_plasma + integral_nb_dr )
                
                if self.crossed0_flag[j] == True:
                    if self.dr_dxi_1[j]<0:
                        #print('need correction')
                        self.dr_dxi_1[j] = np.abs(self.dr_dxi_1[j])
                        self.crossed0_flag[j] = False

                self.r_next[j] = self.r[j] + self.dxi * self.dr_dxi_1[j]
                                
                if self.r_next[j] < 0:
                    self.r_next[j] = np.abs(self.r_next[j])
                    self.crossed0_flag[j] = True
                    
            self.r[:] = self.r_next
            self.dr_dxi_0[:] = self.dr_dxi_1


@njit
def sum_up_to_j( a, j, r_axis ):
    N_r = r_axis.size
    
    sum_result = 0
    
    for l in range(N_r): 
        if (r_axis[l]<=r_axis[j]) and (l!=j):
            sum_result += a[l]    
            
    return sum_result


