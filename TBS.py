'''
with burchard stability and Neumann BC for eps solved with TDMA, u,rho, on i+1/2| k,eps,nu_t on i
'''
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pdb
import re
import scipy.linalg as la
import time
import pandas as pd
import matplotlib.ticker as mtick

start_time = time.time()

## N = number of discretizations in the vertical (this code uses constant cell size)
## DEPTH = vector containing the depth levels at which measurements were made for the point at which the code computes
## ECHODEPTH = echodepth at the point F1
## PRESSGRAD = the pressure gradient between point F1 and F2. Since only lowermost point of F1 and uppermost instrument of F2
##             are at a comparable level, the pressure gradient is computed between these two instrument readings over the time period
##             and then time averaged. Because of lack of data, the press gradient is constant in the vertical.
## RHO = density at point F1, time averaged for the period. This is to be interpolated to model grid and used for the vertical
##       density gradient in the buoyancy term of k equation.
## BOTVEL = bottom velocity, time averaged
## TOPVEL = top velocity, time averaged

class Keps_Solver(object):
    def __init__(self, N, LAT, DEPTH, PRESSGRAD_X, PRESSGRAD_Y, RHO, RESVEL, U10, OUTDIR, OUTFILE):
        self.N = N
        self.H = DEPTH[-1]
        self.depth = DEPTH
        self.rho = RHO
        self.ORESVEL = RESVEL #observed resultant velocity in x direction
        self.lat = LAT        
        self.omega = -2.*np.pi/(3600.*24.)
        self.f = 2. * self.omega * np.sin(np.deg2rad(self.lat))
        self.U10 = U10
        if(U10 < 5):
            self.C_D = 1e-3
        else:
            self.C_D = 2.5*1e-3
        
        self.outfile = OUTFILE
        self.outdir = OUTDIR
        
        self.rho0 = 1e3
        self.g = 9.81
        self.dp_x = PRESSGRAD_X
        self.dp_y = PRESSGRAD_Y
        self.nu = 1e-6
        self.rho_air = 1.225
        
        self.C1eps = 1.44
        self.C2eps = 1.92        
        self.Cmu = .09
        self.Cmu_prime = 0.09/1.3
        self.Cmu0 = 0.5562
        self.sigma_k = 1.0
        self.sigma_eps = 1.08 # value of 2.4 obtained from Burchard textbook, eqn 3.123
        self.sigma_rho = 0.7
        self.kappa = .44
        self.dt = 1.
        
        self.k = np.zeros(N)
        self.eps = np.zeros(N)
        self.u = np.zeros(N-1)
        self.v = np.zeros(N-1)
                
        self.ustar = np.zeros(2)
        self.nu_t = np.zeros(N)
        self.nut_prime = np.zeros(N)
        #Beta phase variables, delete later
        self.Cmuvector = np.zeros(N)
        self.Cmuprimevector = np.zeros(N)
        self.alphaN = np.zeros(N)
        self.alphaM = np.zeros(N)
        
        self.iter = 0
        self.norm_eps = np.ones(N)
        self.norm_u = np.ones(N-1)
        self.norm_v = np.ones(N-1)        
        self.norm_k = np.ones(N)
        self.tol_eps = 1e-6

        self.dy1 = 0.0025 #self.H/1e3 #check this value to see if its within viscous sublayer
        ## y+ = yu*/nu should be within 60
        self.y = np.linspace(self.depth[0]+self.dy1 , self.H-self.dy1 ,N)

        self.dy = self.y[1] - self.y[0]
        self.y_mom = self.y[:-1] + self.dy*0.5
        self.y_mom[0] = self.y[0] + 0.001
        self.y_mom[-1] = self.y[-1] - 0.001
        self.rho_regridded = np.zeros(N)

        self.Nsquared = np.zeros(N)
        self.Msquared = np.zeros(N-1)
        self.pressGrad_x = np.zeros(len(self.y_mom))
        self.pressGrad_y = np.zeros(len(self.y_mom))        

        self.flag_brk = 0
        self.z0 = 0.0015 #arbitrarily set, find a better way to do this

    def interpolateToModelGrid(self):
        rho_interp_func = interpolate.interp1d(self.depth, self.rho)
        self.rho_regridded = rho_interp_func(self.y)

        press_grad_func = interpolate.interp1d(self.depth, self.dp_x)
        self.pressGrad_x = press_grad_func(self.y_mom)

        press_grad_func = interpolate.interp1d(self.depth, self.dp_y)
        self.pressGrad_y = press_grad_func(self.y_mom)
                
        ## spl = interpolate.UnivariateSpline(self.depth, self.rho, k=2)
        ## spl.set_smoothing_factor(0.2)
        ## self.rho_regridded = spl(self.y)

    def guess_uvalues(self):
        u_bot = self.ORESVEL[0]
        u_top = self.ORESVEL[-1]
        #self.u[1:-1] = u_bot - (u_bot - u_top)/(self.H - self.depth[0]) * self.y_mom[1:-1]
        self.u[:] = 0.0

    def guess_eps_values(self):
        self.eps[:] = 1e-8

    def guess_k_values(self):
        self.k[1:-1] = 1e-6 #1e-2*self.y[1:-1]

    def guess_diff_values(self):
        self.nu_t[1:-1] = 1e-4 #self.nu_t[0]

    def set_BC(self):
        z0 = self.z0
        #surface ustar
        self.ustar[-1] = np.sqrt(self.rho_air/self.rho0 * self.C_D ) * self.U10

        #bottom ustar
        self.ustar[0] = self.ORESVEL[0] * self.kappa / np.log(1 + self.y[0]/z0)

        
        self.u[0] = self.ustar[0] * (5.5 + np.log(self.y[0] * abs(self.ustar[0])/self.nu)/self.kappa) #self.ORESVEL[0]
        self.u[-1] =  self.ustar[-1]*(5.5 + np.log(self.dy1 *abs(self.ustar[-1])/self.nu)/self.kappa) # self.ORESVEL[len(self.ORESVEL) - 1] #
        
        self.k[0] = self.ustar[0]**2 / math.sqrt(self.Cmu)
        typ_eps = 1e-11 #typical eps in mid depth
        typ_nut = 1e-6 #typical nu_t set from measured top velocity
        self.k[self.N-1] = self.ustar[1]**2/math.sqrt(self.Cmu) #math.sqrt(typ_eps*typ_nut/self.Cmu) 
        
        self.eps[0] =  self.Cmu0**3 * self.k[0]**1.5 / (self.kappa * self.y[0]) #this variable is solved with a neumann BC, eps[0] is also computed, but this is the initial guess value
        self.eps[-1] = self.Cmu0**3 * self.k[-1]**1.5 / (self.kappa * self.dy1) #self.u_star[1]**3/(self.kappa * self.dy1) #typ_eps ##
        
        self.nu_t[0] = self.kappa * self.ustar[0] * self.y[0] #self.Cmu*self.k[0]**2/self.eps[0]
        self.nu_t[-1] = self.kappa * self.ustar[-1] * self.dy1 #self.kappa * self.ustar[1] * self.dy1 #self.Cmu * self.k[self.N-1]**2/self.eps[self.N-1] #typ_nut #

        self.nut_prime[0] = self.nu_t[0]/self.sigma_rho
        self.nut_prime[self.N-1] = self.nu_t[self.N-2]/self.sigma_rho

    
    def computeStabilityFunction(self):
        ## Following model constants and coefficients were obtained from Burchard and Petersen, 1999
        Cmu0 = 0.5562 #from table 1
        c1 = 5.
        c2 = 0.8
        c3 = 1.968
        c4 = 1.136
        c5 = 0.
        c6 = 0.4

        cb1 = 5.95
        cb2 = 0.6
        cb3 = 1.
        cb4 = 0.
        cb5 = 0.3333
        cbb = 0.72

        a1 = 2./3. - 0.5*c2
        a2 = 1.-0.5*c3
        a3 = 1. - 0.5*c4
        a4 = 0.
        a5 = 0.5 - 0.5*c6

        ab1 = 1-cb2
        ab2 = 1-cb3
        ab3 = 2.*(1.-cb4)
        ab4 = 2.*(1.-cb5)
        ab5 = 2*cbb*(1.-cb5)
        
        N = c1*0.5 # production to dissipation ratio (P+G)/eps
        Nb = cb1

        d0 = 36. * N**3 * Nb**2
        d1 = 84.*a5*ab3*N**2*Nb + 36*ab5*N**3*Nb
        d2 = 9*(ab2**2 - ab1**2)*N**3 - 12.*(a2**2 - 3*a3**2)*N*Nb**2
        d3 = 12.*a5* ab3* (a2*ab1 - 3.*a3*ab2)*N + 12.*a5*ab3*(a3**2 - a2**2)* Nb + 12.*ab5*(3*a3**2 - a2**2)*N*Nb
        d4 = 48.*a5**2 * ab3**2 * N + 36.*a5*ab3*ab5* N**2
        d5 = 3*(a2**2 - 3.*a3**2)*(ab1**2 - ab2**2)*N
        n0 = 36.*a1* N**2 * Nb**2
        n1 = -12.*a5*ab3*(ab1 + ab2)*N**2 +  8.*a5*ab3*(6*a1 - a2 - 3*a3)*N*Nb +  36.*a1*ab5*N**2* Nb
        n2 = 9.*a1 *(ab2**2 - ab1**2)*N**2
        nb0 = 12.*ab3**2 * N**3 *Nb
        nb1 = 12.* a5*ab3**2 * N**2
        nb2 = 9.*a1* ab3*(ab1 - ab2)*N**2 + (6.*a1 *(a2 - 3*a3) - 4.*(a2**2 - 3.*a3**2) ) *ab3 *N*Nb

        # Numerical computations follow:
        alphaN = np.zeros(self.N)
        alphaNtilde = np.zeros(self.N)
        alphaN[:] = (self.k/self.eps)**2 * self.Nsquared
        alphaC = -0.02
        alphamin = -0.0466
        alphamax = 0.56
        alphaNcut = alphaN - (alphaN - alphaC)**2 / (alphaN + alphamin - 2*alphaC)
        alphaNtilde[:] = np.maximum(alphaN[:], alphaNcut[:])
        alphaNtilde[(alphaNtilde > alphamax)] = alphamax

        num = d0*n0 + (d0*n1+d1*n0)*alphaNtilde + (d1*n1+d4*n0)*alphaNtilde**2 + d4*n1*alphaNtilde**3
        den = d2*n0 + (d2*n1 + d3*n0)*alphaNtilde + d3*n1*alphaNtilde**2

        alphaM_max = num/den

        ## if(self.iter > 2):
        ##     alphaM[:] = (1 + self.Cmuprimevector*alphaNtilde)/self.Cmuvector
        ## else:
        ##     du = np.zeros(self.N)
        ##     dz = np.zeros(self.N)
        ##     du[1:-1] = self.u[1:] - self.u[:-1]
        ##     du[0] = self.u[1] - self.u[0]
        ##     du[-1] = self.u[-1] - self.u[-2]
            
        ##     dz[:-1] = self.y[1:] - self.y[:-1]
        ##     dz[-1] = self.y[-1] - self.y[-2]
            
        ##     Su = self.k/self.eps * du/dz
        msq_avg = np.zeros(self.N)
        msq_avg[1:-1] = (self.Msquared[:-1] + self.Msquared[1:]) * 0.5
        msq_avg[0] = self.Msquared[0]
        msq_avg[-1] = self.Msquared[-1]
        
        alphaM = (self.k/self.eps)**2 * msq_avg
        alphaM[(alphaM[:] > alphaM_max[:])] = alphaM_max[(alphaM[:] > alphaM_max[:])]
        
        num = n0 + n1*alphaNtilde + n2*alphaM
        num_prime = nb0 + nb1*alphaNtilde + nb2*alphaM
        den = d0 + d1*alphaNtilde + d2*alphaM + d3*alphaNtilde*alphaM + d4*alphaNtilde**2 + d5*alphaM**2
        self.Cmuvector[:] = num/den
        self.Cmuprimevector[:] = num_prime/den

        coeff = (d1 + d3*alphaM + 2.*d4*alphaNtilde)/(alphaNtilde * den) + 1./(alphaNtilde**2)
        limiter = nb1/(alphaNtilde * den * coeff)
        if(self.flag_brk == 1 ): #or self.iter > 1044):
            pdb.set_trace()
            
        #self.Cmuprimevector[self.Cmuprimevector < limiter] = limiter[self.Cmuprimevector < limiter]
        
        self.alphaN[:] = alphaNtilde
        self.alphaM[:] = alphaM
        
    
    def computeMsquared(self):
        self.Msquared[1:] = ( (self.u[1:] - self.u[:-1]) / (self.y_mom[1:] - self.y_mom[:-1]) )**2 + ( (self.v[1:] - self.v[:-1]) / (self.y_mom[1:] - self.y_mom[:-1]) )**2
        self.Msquared[0] = ( (self.u[1] - self.u[0]) / (self.y_mom[1] - self.y_mom[0]) )**2 + ( (self.v[1] - self.v[0]) / (self.y_mom[1] - self.y_mom[0]) )**2
    
    def computeBuoyancy(self):
        self.Nsquared[1:] = -self.g/self.rho0 * (self.rho_regridded[1:] -
                                             self.rho_regridded[:-1]) / (self.y[1:] - self.y[:-1] )
        self.Nsquared[0] = -self.g/self.rho0 * (self.rho_regridded[1] -
                                           self.rho_regridded[0]) / (self.y[1] - self.y[0])

    def computeDiffusivities(self):
        #self.nu_t[:-1] = self.Cmuvector[:-1] * self.k[:-1]**2 / self.eps[:-1] # * self.Cmu0**3
        #self.nut_prime[:-1] = self.Cmuprimevector[:-1] * self.k[:-1]**2 / self.eps[:-1] # * self.Cmu0**3
        k_avg = (self.k[:-2] + self.k[2:])*0.5
        eps_avg = (self.eps[:-2] + self.eps[2:])*0.5
        self.nu_t[:] = self.Cmuvector[:] * self.k[:]**2 / self.eps[:] #(k_avg)**2 / eps_avg  #* self.Cmu0**3
        self.nu_t[0] = self.Cmuvector[0] * (self.k[0])**2 / self.eps[0]  * self.Cmu0**3
        self.nu_t[-1] = self.Cmuvector[-1] * (self.k[-1])**2 / self.eps[-1]  #* self.Cmu0**3
        
        #self.nu_t[self.nu_t > 1e-3] = 1e-3
        self.nut_prime[:] = self.Cmuprimevector[:] * self.k**2/self.eps #k_avg**2 / eps_avg  #* self.Cmu0**3
        self.nut_prime[0] = self.Cmuprimevector[0] * (self.k[0])**2 / self.eps[0]  * self.Cmu0**3
        self.nut_prime[-1] = self.Cmuprimevector[-1] * (self.k[-1])**2 / self.eps[-1]  #* self.Cmu0**3
        #self.nut_prime[self.nut_prime > 1e-3] = 1e-3

    def update_ustar(self):
        #bottom ustar
        self.ustar[0] = self.kappa / np.log(self.y_mom[0]/self.z0) * np.sqrt(self.u[0]**2 + self.v[0]**2)
        self.z0 = 0.1*self.nu/self.ustar[0] + 0.03*self.y_mom[0]

    def compute_uv_2Neum(self):
        ab_u = np.zeros((3, self.N-1))
        ab_v = np.zeros((3, self.N-1))
        dvector_u = np.zeros(self.N-1)
        dvector_v = np.zeros(self.N-1)
        
        u_old = np.copy(self.u)
        v_old = np.copy(self.u)        
        N = self.N

        dz = np.zeros(self.N-1)
        dz[:-1] = self.y_mom[1:] - self.y_mom[:-1]
        dz[-1] = self.y_mom[-1] - self.y_mom[-2]
        nu_avg = (self.nu_t[1:] + self.nu_t[:-1]) *0.5 + self.nu
        cstar = self.kappa / (np.log(1 + self.y_mom[0]/self.z0))
        # coeff_{i-1} should have trailing zeros
        ab_u[2, :-1] = -nu_avg[:-1] / dz[:-1]**2
        ab_u[2, -1] = 0.

        ab_v[2, :-1] = -nu_avg[:-1] / dz[:-1]**2
        ab_v[2, -1] = 0.

        coeff_im1 = np.append(ab_u[2], ab_v[2])
        

        # u_i coeff should have all diagonal terms
        ab_u[1, 1:-1] = 1./self.dt + 2 * nu_avg[1:-1]/dz[1:-1]**2
        ab_u[1, 0] = 1./ self.dt + nu_avg[0]/dz[0]**2 + cstar**2 * u_old[0]/dz[0]
        ab_u[1, -1] = 1./self.dt + nu_avg[-1]/dz[-1]**2

        ab_v[1, 1:-1] = 1./self.dt + 2 * nu_avg[1:-1]/dz[1:-1]**2
        ab_v[1, 0] = 1./ self.dt + nu_avg[0]/dz[0]**2 + cstar**2 * v_old[0]/dz[0]
        ab_v[1, -1] = 1./self.dt + nu_avg[-1]/dz[-1]**2 
        
        coeff_i = np.append(ab_u[1], ab_v[1])
        
        # u_{i+1} should have all leading zeros
        ab_u[0, 1:] = -nu_avg[1:] / dz[1:]**2
        ab_u[0, 0] = 0.

        ab_v[0, 1:] = -nu_avg[1:] / dz[1:]**2
        ab_v[0, 0] = 0.

        coeff_ip1 = np.append(ab_u[0], ab_v[0])

        dvector_u[:] = -self.pressGrad_x / self.rho0 + self.f * self.v + u_old/self.dt
        dvector_u[-1] += self.ustar[-1]**2/dz[-1]
        
        dvector_v[:] = -self.pressGrad_y / self.rho0 - self.f * self.u + v_old/self.dt
        dvector_v[-1] += self.ustar[-1]**2/dz[-1]

        dvector = np.append(dvector_u, dvector_v)
        
        uv = la.solve_banded((1,1), np.array((coeff_ip1, coeff_i, coeff_im1)), dvector)
        self.u[:] = uv[0:N-1]
        self.v[:] = uv[N-1:]
        self.norm_u[:] = abs(u_old[:] - self.u[:])/abs(self.u[:])
        self.norm_v[:] = abs(v_old[:] - self.v[:])/abs(self.v[:])        
        del u_old, v_old
        
    def compute_u_2Neum(self):
        ab = np.zeros((3, self.N-1))
        dvector = np.zeros(self.N-1)
        u_old = np.copy(self.u)
        N = self.N

        dz = np.zeros(self.N-1)
        dz[:-1] = self.y_mom[1:] - self.y_mom[:-1]
        dz[-1] = self.y_mom[-1] - self.y_mom[-2]
        nu_avg = (self.nu_t[1:] + self.nu_t[:-1]) *0.5 + self.nu
        cstar = self.kappa / (np.log(1 + self.y_mom[0]/self.z0))
        # u_{i-1} should have trailing zeros
        ab[2, :-1] = -nu_avg[:-1] / dz[:-1]**2
        ab[2, -1] = 0.

        # u_i coeff should have all diagonal terms
        ab[1, 1:-1] = 1./self.dt + 2 * nu_avg[1:-1]/dz[1:-1]**2
        ab[1, 0] = 1./ self.dt + nu_avg[0]/dz[0]**2 + cstar**2 * u_old[0]/dz[0]
        ab[1, -1] = 1./self.dt + nu_avg[-1]/dz[-1]**2 

        # u_{i+1} should have all leading zeros
        ab[0, 1:] = -nu_avg[1:] / dz[1:]**2
        ab[0, 0] = 0.

        dvector[:] = -self.pressGrad_x / self.rho0 + self.f * self.v + u_old/self.dt
        dvector[-1] += self.ustar[-1]**2/dz[-1]
        self.u[:] = la.solve_banded((1,1), ab, dvector)
        self.norm_u[:] = abs(u_old[:] - self.u[:])/abs(self.u[:])
        del u_old

    def compute_v_2neum(self):
        ab = np.zeros((3, self.N-1))
        dvector = np.zeros(self.N-1)
        v_old = np.copy(self.u)
        N = self.N
        
        dz = np.zeros(self.N-1)
        dz[:-1] = self.y_mom[1:] - self.y_mom[:-1]
        dz[-1] = self.y_mom[-1] - self.y_mom[-2]
        nu_avg = (self.nu_t[1:] + self.nu_t[:-1]) *0.5 + self.nu
        cstar = self.kappa / (np.log(1 + self.y_mom[0]/self.z0))        
        # u_{i-1} should have trailing zeros
        ab[2, :-1] = -nu_avg[:-1] / dz[:-1]**2
        ab[2, -1] = 0.

        # u_i coeff should have all diagonal terms
        ab[1, 1:-1] = 1./self.dt + 2 * nu_avg[1:-1]/dz[1:-1]**2
        ab[1, 0] = 1./ self.dt + nu_avg[0]/dz[0]**2 + cstar**2 * v_old[0]/dz[0]
        ab[1, -1] = 1./self.dt + nu_avg[-1]/dz[-1]**2 

        # u_{i+1} should have all leading zeros
        ab[0, 1:] = -nu_avg[1:] / dz[1:]**2
        ab[0, 0] = 0.

        dvector[:] = -self.pressGrad_y / self.rho0 - self.f * self.u + v_old/self.dt
        dvector[-1] += self.ustar[-1]**2/dz[-1]
        self.v[:] = la.solve_banded((1,1), ab, dvector)
        self.norm_v[:] = abs(v_old[:] - self.v[:])/abs(self.v[:])
        del v_old
        
    def compute_u(self):
        ab = np.zeros((3,self.N-3))
        dvector = np.zeros(self.N-3)
        u_old = np.copy(self.u)
        omega = -2.*np.pi/(3600.*24.)
        f = 2. * omega * np.sin(np.deg2rad(self.lat))
        N = self.N
        
        dz = np.zeros(self.N-1)
        dz[:-1] = self.y_mom[1:] - self.y_mom[:-1]
        dz[-1] = self.y_mom[-1] - self.y_mom[-2]

        # u_{i-1} coefficient, should have trailing zeros
        nu_avg_a = (self.nu_t[:N-3] + self.nu_t[1:N-2])*0.5 + self.nu
        ab[2,:] = - nu_avg_a / dz[:-2]**2
        ab[2,-1] = 0.

        # u_i coeff, should have all the diagonal terms
        nu_avg_b = (self.nu_t[1:N-2] + self.nu_t[2:N-1]) * 0.5 + self.nu
        ab[1,:] = 2. * nu_avg_b / dz[1:-1]**2 + 1./self.dt

        # u_{i+1} coeff, should have leading zeros
        nuavg_c = (self.nu_t[2:N-1] + self.nu_t[3:N] )*0.5 + self.nu
        ab[0,:] = -nuavg_c / dz[2:]**2
        ab[0,0] = 0.

        dvector[:] = -self.pressGrad[1:-1] / self.rho0 + u_old[1:-1] / self.dt + f*self.v[1:-1]
        dvector[0] = dvector[0] + self.u[0] * nu_avg_a[0] / dz[0]**2
        dvector[-1] = dvector[-1] + self.u[-1] * nuavg_c[-1] / dz[-1]**2

        self.u[1:-1] = la.solve_banded((1,1), ab, dvector)
        
        self.norm_u[:] = abs(u_old[1:-1] - self.u[1:-1])/abs(self.u[1:-1])        
        del u_old

    def compute_k_values_2N(self):
        ab = np.zeros((3,self.N))
        dvector = np.zeros(self.N)
        N = self.N
        
        nu_k = self.nu_t[:] / self.sigma_k
        k_old = np.copy(self.k)
        dz = np.zeros(self.N)
        dz[:-1] = abs(self.y[1:] - self.y[:-1])
        dz[-1] = abs(self.y[-1] - self.y[-2])

        buoy = np.zeros(N)
        prod = np.zeros(N)
        stable = (self.Nsquared[:] > 0.) # mask/filter to get bool for indices where stable is True
        unstable = np.invert(stable)
        buoy[stable] = -(self.nut_prime[stable]  * self.Nsquared[stable] / k_old[stable] )
        buoy[unstable] = - self.nut_prime[unstable] * self.Nsquared[unstable]

        # setting k_{i-1} coefficients, trailing zeros
        ab[2,:-1] = -nu_k[:-1] / dz[:-1]**2
        ab[2,-1] = 0.
        # setting k_{i} coeff, diagonal terms
        ab[1,1:-1] = 2.* nu_k[1:-1]/dz[1:-1]**2 + 2. * self.eps[1:-1]/k_old[1:-1] + 1./self.dt
        ab[1,0] = 1.* nu_k[0]/dz[0]**2 + 2. * self.eps[0]/k_old[0] + 1./self.dt
        ab[1, -1] = 1.* nu_k[-1]/dz[-1]**2 + 2. * self.eps[-1]/k_old[-1] + 1./self.dt
        ab[1, stable] = ab[1,stable] - buoy[stable]

        #setting k_{i+1} coeffs, leading zeros
        ab[0,1:] = -nu_k[1:]/dz[1:]**2
        ab[0,0] = 0.

        msq_avg = np.zeros(N)
        msq_avg[1:-1] = (self.Msquared[:-1] + self.Msquared[1:]) * 0.5
        msq_avg[0] = self.Msquared[0]
        msq_avg[-1] = self.Msquared[-1]
        
        prod[:] = self.nu_t[:] * msq_avg[:]

        dvector = prod + self.eps[:] + k_old[:] / self.dt
        dvector[unstable] = dvector[unstable] + buoy[unstable]

        ## if( (abs(ab[0]) > abs(ab[1])).any() | (abs(ab[2]) > abs(ab[1])).any() ):
        ##     pdb.set_trace()
        ##     self.flag_brk = 1
        self.k[:] = la.solve_banded((1,1), ab, dvector)

        self.k[np.where(self.k[:] < 3e-8)] = 3e-8
            
        self.norm_k[:] = abs(k_old[:] - self.k[:])/abs(self.k[:])

    
    def compute_k_values(self):
        ab = np.zeros((3,self.N-2))
        dvector = np.zeros(self.N-2)
        N = self.N
        
        nu_k = self.nu_t[:] / self.sigma_k
        k_old = np.copy(self.k)
        dz = np.zeros(self.N)
        dz[:-1] = self.y[1:] - self.y[:-1]
        dz[-1] = self.y[-1] - self.y[-2]

        buoy = np.zeros(N-2)
        prod = np.zeros(N-2)
        stable = (self.Nsquared[1:-1] > 0.)
        unstable = np.invert(stable)
        buoy[stable] = (self.nut_prime[1:-1][stable] / self.sigma_k * self.Nsquared[1:-1][stable]
                        / k_old[1:-1][stable] )
        buoy[unstable] = self.nut_prime[1:-1][unstable] * self.Nsquared[1:-1][unstable]

        # setting k_{i-1} coefficients
        ab[2] = nu_k[:-2] / dz[:-2]**2
        ab[2,-1] = 0.
        # setting k_{i} coeff
        ab[1] = -2.* self.nu_t[1:-1]/dz[1:-1]**2 -2. * self.eps[1:-1]/k_old[1:-1] - 1./self.dt
        ab[1, stable] = ab[1,stable] - buoy[stable]

        #setting k_{i+1} coeffs
        ab[0] = nu_k[2:]/dz[2:]**2
        ab[0,0] = 0.

        msq_avg = np.zeros(len(self.Msquared))
        msq_avg[:-1] = self.Msquared[:-1] + self.Msquared[1:]
        msq_avg[-1] = (self.Msquared[-1] + self.Msquared[-2]) * 0.5
        
        prod[:] = self.nu_t[1:-1] * msq_avg[:-1]

        dvector = -prod - self.eps[1:-1] - k_old[1:-1] / self.dt
        dvector[unstable] = dvector[unstable] + buoy[unstable]
        dvector[0] = dvector[0] - nu_k[0] / dz[0]**2 * self.k[0]
        dvector[-1] = dvector[-1] - nu_k[-1]/dz[-1]**2 * self.k[-1]

        if( (abs(ab[0]) > abs(ab[1])).any() | (abs(ab[2]) > abs(ab[1])).any() ):
            pdb.set_trace()
            self.flag_brk = 1

        self.k[1:-1] = la.solve_banded((1,1), ab, dvector)

        self.k[np.where(self.k[1:-1] < 3e-8)] = 3e-8
            
        self.norm_k[:] = abs(k_old[1:-1] - self.k[1:-1])/abs(self.k[1:-1])

    
    def compute_eps_td_patankar_1N(self): #Neumann BC solver
        a = np.zeros(self.N-1)
        b = np.zeros(self.N-1)
        c = np.zeros(self.N-1)
        d = np.zeros(self.N-1)
        nu_tot =  self.nu_t[:]/self.sigma_eps
        nu_eps = self.nu_t[:]/self.sigma_eps        
        Ceps1 = 1.44
        Ceps2 = 1.92
        Ceps3 = 0.
        Cmu0 = 0.5562        
        oldeps = np.copy(self.eps)
        N = self.N
        
        dz = np.zeros(N)
        dz[:-1] = self.y[1:] - self.y[:-1]
        dz[-1] = self.y[-1] - self.y[-2]
        
        Ceps3 = np.zeros(N-1)
        Ceps3[(self.Nsquared[:-1] < 0.)] = 1.0
        Ceps3[(self.Nsquared[:-1] > 0.)] = -0.4

        coeff_nl_lhs = -Ceps2 * 2. * oldeps[:-1] / self.k[:-1]
        coeff_nl_rhs = Ceps2 * oldeps[:-1]**2 / self.k[:-1]

        buoy = np.zeros(N-1)
        stable = (Ceps3 * self.Nsquared[:-1] > 0.)
        unstable = np.invert(stable)
        buoy[stable] = (-Ceps3[stable] * self.Cmuprimevector[:-1][stable] * self.Nsquared[:-1][stable]
                        * self.k[:-1][stable] / oldeps[:-1][stable] )
        buoy[unstable] = (-Ceps3[unstable] * self.Cmuprimevector[:-1][unstable] *
                          self.k[:-1][unstable] * self.Nsquared[:-1][unstable] )
        # computing coeff_i
        coeff_i = np.zeros(N-1)
        coeff_i[1:] = -2. * nu_eps[1:-1] / dz[1:-1]**2 + coeff_nl_lhs[1:]
        coeff_i[0] = -nu_eps[0] / dz[0]**2 + coeff_nl_lhs[0]
        coeff_i[stable] = coeff_i[stable] + buoy[stable] - 1./self.dt
        rfict = nu_eps[0] / dz[0] * self.k[0]**(1.5) / (self.kappa * self.y[0]**2) # * self.Cmu0**3 

        coeff_im1 = np.zeros(N-1)
        coeff_im1[:-1] = nu_eps[:-2] / dz[:-2]**2
        coeff_im1[-1] = 0.

        coeff_ip1 = np.zeros(N-1)
        coeff_ip1[1:] = nu_eps[1:-1] / dz[1:-1]**2
        coeff_ip1[0] = 0.

        prod = np.zeros(N-1)
        msq_avg = np.zeros(N)
        msq_avg[1:-1] = (self.Msquared[1:] + self.Msquared[:-1])*0.5
        msq_avg[0] = self.Msquared[0]
        msq_avg[-1] = self.Msquared[-1]
        
        prod[:] = Ceps1 * self.Cmuvector[:-1] * self.k[:-1] * msq_avg[:-1]
        bvector = np.zeros(N-1)
        bvector[:] = -prod[:] - coeff_nl_rhs[:] - oldeps[:-1]/self.dt
        bvector[0] = bvector[0] - rfict
        bvector[-1] = -nu_eps[-1] * self.eps[-1]/dz[-1]**2
        bvector[unstable] = bvector[unstable] - buoy[unstable]

        if( (abs(coeff_im1) > abs(coeff_i)).any() | (abs(coeff_ip1) > abs(coeff_i) ).any() ):
            pdb.set_trace()
            self.flag_brk = 1

        self.eps[:-1] = la.solve_banded((1,1), np.array((coeff_ip1, coeff_i, coeff_im1)), bvector )
        self.norm_eps[:-1] = abs(self.eps[:-1] - oldeps[:-1])/abs(self.eps[:-1].max())
        del oldeps

    def compute_eps_td_patankar_2N(self): # Top and bottom Neumann BC solver
        dvector = np.zeros(self.N)
        
        nu_eps = self.nu_t[:]/self.sigma_eps
        
        Ceps1 = 1.44
        Ceps2 = 1.92
        Ceps3 = np.zeros(len(self.Nsquared))
        Cmu0 = 0.5562
        
        oldeps = np.copy(self.eps)
        N = self.N
        
        Ceps3[(self.Nsquared < 0.)] = 1. #if unstable strat
        Ceps3[(self.Nsquared > 0.)] = -0.4 #if stable strat
        dz = np.zeros(N)
        dz[:-1] = self.y[1:] - self.y[:-1]
        dz[-1] = self.y[-1] - self.y[-2]
        

        coeff_i = np.zeros(N)
        coeff_im1 = np.zeros(len(self.eps))
        coeff_ip1 = np.zeros(len(self.eps))
        
        coeff_nl_lhs = np.zeros(len(self.eps))
        coeff_nl_rhs = np.zeros(len(self.eps))
        
        buoy = np.zeros(N)
        
        prod = np.zeros(N)
        msq_avg = np.zeros(N)
        
        pat = (Ceps3 * self.Nsquared > 0) # if the sink term is positive, do patankar type discretization
        nopat = np.invert(pat) # else, leave sink term on the rhs, ie. as is
        buoy[pat] = Ceps3[pat] * self.Cmuprimevector[pat]  * self.Nsquared[pat] * self.k[pat]/oldeps[pat] # * self.Cmu0**3
        buoy[nopat] = -Ceps3[nopat] * self.Cmuprimevector[nopat] * self.k[nopat] * self.Nsquared[nopat] # *self.Cmu0**3
        
        coeff_nl_lhs = Ceps2 * 2.* oldeps / self.k
        coeff_nl_rhs = Ceps2 * oldeps**2 / self.k
        
        #computing coeff_i
        coeff_i[1:-1] = 2. * nu_eps[1:-1] / dz[1:-1]**2 + coeff_nl_lhs[1:-1] + 1/self.dt
        coeff_i[0] = nu_eps[0] / dz[0]**2 + coeff_nl_lhs[0] + 1/self.dt
        rfict0 = -nu_eps[0]/dz[0] * self.k[0]**(1.5)/(self.kappa * self.y[0]**2) * self.Cmu0**3
        coeff_i[N-1] = nu_eps[N-1] / dz[N-1]**2 + coeff_nl_lhs[N-1] + 1/self.dt
        rfictN = nu_eps[N-1] / dz[N-1] * self.k[N-1]**(1.5)/(self.kappa * self.dy1**2) * self.Cmu0**3
        coeff_i[pat] += buoy[pat]

        #computing coeff_im1, trailing zeros
        coeff_im1[:-1] = -nu_eps[:-1] / dz[:-1]**2
        coeff_im1[-1] = 0.0

        #computing coeff_ip1, leading zeros
        coeff_ip1[1:] = -nu_eps[1:] / dz[1:]**2
        coeff_ip1[0] = 0.0
        
        msq_avg[1:-1] = (self.Msquared[1:] + self.Msquared[:-1])*0.5
        msq_avg[0] = self.Msquared[0]
        msq_avg[-1] = self.Msquared[-1]

        prod = Ceps1 * self.Cmuvector * self.k * msq_avg # * Cmu0**3
        
        dvector = prod + coeff_nl_rhs + oldeps/self.dt
        dvector[0] += rfict0
        dvector[N-1] += rfictN
        dvector[nopat] += buoy[nopat]

        ## if( (abs(coeff_im1) > abs(coeff_i)).any() | (abs(coeff_ip1) > abs(coeff_i)).any() ):
        ##     pdb.set_trace()
        ##     self.flag_brk = 1
        
        self.eps[:] = la.solve_banded((1,1), np.array((coeff_ip1, coeff_i, coeff_im1)) ,dvector)
        mask1 = self.Nsquared > 0
        limiter = 0.045 * self.k**2 * self.Nsquared
        mask2 = self.eps**2 < limiter
        #pdb.set_trace()        
        self.eps[mask1 & mask2] = np.sqrt(limiter[mask1 & mask2])
        
        self.norm_eps[:] = abs(self.eps[:] - oldeps[:])/abs(self.eps[:].max())
        del oldeps
        
    def WriteOutput(self):
        if(os.path.isdir(str(self.outdir)) == False):
            os.system('mkdir '+str(self.outdir))
        time = int(self.iter*self.dt)
        filename = str(self.outdir)+str(self.outfile)+str(time)
        with open(str(filename), 'w') as outfile:
            tab = str('\t')
            outfile.write('N='+str(self.N)+'\n')
            outfile.write('mab \t U \t k \t eps \t nu_t \n')
            for i in range(self.N):
                if(i < self.N-1):
                    outfile.write(str(self.y[i])+tab+str(self.u[i])+tab+str(self.k[i])+tab+str(self.eps[i])+tab+str(self.nu_t[i])+str('\n'))
                else:
                    outfile.write(str(self.y[i])+tab+str(1e-44)+tab+str(self.k[i])+tab+str(self.eps[i])+tab+str(self.nu_t[i])+str('\n'))
        
    def plotter(self, mode='single', wd=5, ht=5):
        dv = 100           
        time = self.iter*self.dt
        fspath = str(self.outdir)

        fig, ax = plt.subplots()
        #ax.figure(figsize=(wd,ht))
        ax.set_title('U vs Depth')
        ax.set_xlabel('$ms^{-1}$')
        ax.set_ylabel('mab')
        ax.grid(True)
        #plt.xlim(np.mean(self.u)-5*np.std(self.u), np.mean(self.u)+5*np.std(self.u))
        #ax.set_xlim(-10,10)
        ax.set_ylim(self.depth[0]-10,self.H+10)
        ax.plot(self.u,self.y_mom,label = 't='+str(time)+str("s"))
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(fspath)+str('U')+str(time)+str('.png'), dpi=dv, bbox_inches='tight')

        plt.figure(figsize=(wd,ht))
        plt.xlabel('$m^2s^{-2}$')
        plt.ylabel('mab')
        plt.title('K vs Depth')
        plt.grid(True)
        plt.ylim(self.depth[0]-10,self.H+10)
        plt.xscale("log")
        plt.xlim(1e-8, 1e-1)
        plt.plot(self.k,self.y,label = 't='+str(time)+str("s"))
        plt.legend()
        plt.tight_layout()        
        plt.savefig(str(fspath)+str('k')+str(time)+str('.png'), dpi=dv)

        plt.figure(figsize=(wd,ht))
        plt.title(r'$\epsilon$ vs Depth')
        plt.xlabel('$m^2s^{-3}$')
        plt.ylabel('mab')
        plt.grid(True)
        plt.xscale("log")
        plt.xlim(1e-11,1e1)
        plt.ylim(self.depth[0]-10,self.H+10)
        plt.plot(self.eps,self.y,label = r't='+str(time)+str("s"))
        plt.legend()
        plt.tight_layout()        
        plt.savefig(str(fspath)+str('eps')+str(time)+str('.png'), dpi=dv)


        plt.figure(figsize=(wd,ht))
        plt.title(r'$\mathcal{V}_t$ vs Depth')
        plt.ylabel('mab')
        plt.xlabel('$m^2s^{-1}$')
        plt.grid(True)
        plt.ylim(self.depth[0]-10,self.H+10)
        plt.xscale("log")
        plt.xlim(1e-8, 1e-2)        
        plt.plot(self.nu_t,self.y,label = r't='+str(time)+str("s"))
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(fspath)+str('nut')+str(time)+str('.png'), dpi=dv, frameon=True)
        
        if(mode == "single"):
            plt.close("all")
        elif(mode == "cumulative"):
            pass
        else:
            return 0
            
    def readOutput(self, filename):
        with open(str(filename), 'r') as infile:
            counter = 0
            for line in infile:
                m = re.search('[A-Z]', line)

                if(m == None):
                    self.y[counter] = float(line.split()[0])
                    if(counter < self.N-1):
                        self.u[counter] = float(line.split()[1])
                    self.k[counter] = float(line.split()[2])
                    self.eps[counter] = float(line.split()[3])
                    self.nu_t[counter] = float(line.split()[4])
                    counter += 1

                elif(m.group() == "N"):
                    self.N = float(line.split("=")[1])

    def solver(self, MAXITER, DT, SAVE_INTERVAL):
        self.interpolateToModelGrid()
        #self.set_BC()
        self.guess_uvalues()
        self.guess_eps_values()
        self.guess_k_values()
        self.guess_diff_values()
        self.set_BC()
        self.computeBuoyancy()
        self.iter = 0
        self.dt = float(DT)
        save_int = SAVE_INTERVAL
        leastiter = 0
        maxiter = MAXITER
        maxnormk = np.zeros(maxiter)
        self.computeMsquared()
        tol = 1e-3
        while(self.iter < maxiter ):
             #& (max(self.norm_k) > tol or max(self.norm_eps) > tol or max(self.norm_u) > tol)): #self.iter < maxiter
            self.computeStabilityFunction()
            self.compute_uv_2Neum()
            #self.compute_v_2neum()
            self.compute_eps_td_patankar_2N()
            self.compute_k_values_2N()
            self.computeMsquared()
            self.computeDiffusivities()
            self.update_ustar()
            self.iter += 1
            #print("this is iteration number = "+str(self.iter)+", max(norm_k)= "+str(max(self.norm_k))+", norm_eps_max="+str(max(self.norm_eps[:-1]))+", norm_u_max="+str(max(self.norm_u)))
            if(save_int != 0):
                if(self.iter % save_int == 0):
                    self.WriteOutput()
                    self.plotter("single")
