'''
fullyCoupled branch1: to be merged with fullyCoupled, this version has nu_t used for ustar and yplus instead of nu
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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
    def __init__(self, LAT, DEPTH, PRESSGRAD_X, PRESSGRAD_Y, RHO, U10, V10, OUTDIR, OUTFILE, init_u=[0.1,0.1],init_v=[0.1,0.1], dy=0.25, mode="quiet", dy1_bot=0.05, dy1_surf=0.1, z0_bot=0.0012, case="None Provided"):

        self.H = DEPTH[-1]
        self.dy = dy        
        N = int(self.H/self.dy)
        self.depth = DEPTH
        self.rho = RHO
        self.init_u = init_u # initial guess of u velocity, bottom and top
        self.init_v = init_v
        self.lat = LAT        
        self.omega = 2.*np.pi/(3600.*24.)
        self.f = 2. * self.omega * np.sin(np.deg2rad(self.lat))
        self.U10 = U10
        self.V10 = V10
        self.case = str(case)

        #############################################33 Stretch Grid
        #############################################
        no_zoom = 20
        self.z0 = z0_bot #arbitrarily set, find a better way to do this        
        self.dy1_bot = dy1_bot #self.H/1e3 #check this value to see if its within viscous sublayer
        ## y+ = yu*/nu should be within 60
        self.dy1_surf = dy1_surf
        
        self.N = N
        self.y = np.linspace(self.depth[0]+self.dy1_bot+self.z0 , self.H-self.dy1_surf , self.N)
        self.dy = self.y[1] - self.y[0]
        self.y_mom = self.y[:-1] + self.dy*0.5
        
        dz_zoom = (self.y[1] - self.y[0]) / no_zoom
        self.y = np.insert(self.y, 1, np.linspace(self.y[0]+dz_zoom, self.y[1]-dz_zoom, no_zoom))

        dz_zoom = (self.y[-1] - self.y[-2]) / no_zoom
        self.y = np.insert(self.y, len(self.y)-1, np.linspace(self.y[-2]+dz_zoom, self.y[-1]-dz_zoom, no_zoom))

        dy = self.y[1:] - self.y[:-1]
        self.y_mom = self.y[:-1] + dy

        self.N = len(self.y)
        N = self.N
        #################################################
        ##############################################333
                
        if(U10 < 5):
            self.C_Du = 1e-3
        else:
            self.C_Du = 2.5*1e-3
            
        if(V10 < 5):
            self.C_Dv = 1e-3
        else:
            self.C_Dv = 2.5*1e-3
        
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
        self.vstar = np.zeros(2)
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
        self.tol = 1e-5

        ## self.z0 = z0_bot #arbitrarily set, find a better way to do this        
        ## self.dy1_bot = dy1_bot #self.H/1e3 #check this value to see if its within viscous sublayer
        ## ## y+ = yu*/nu should be within 60
        ## self.dy1_surf = dy1_surf
        ## self.N = N
        ## self.y = np.linspace(self.depth[0]+self.dy1_bot+self.z0 , self.H-self.dy1_surf , self.N)
        ## self.dy = self.y[1] - self.y[0]
        ## self.y_mom = self.y[:-1] + self.dy*0.5
        
        #self.y_mom[0] = self.y[0] + (self.dy1_bot + self.z0)*0.5
        #self.y_mom[-1] = self.y[-1] - self.dy1_surf*0.5

        self.rho_regridded = np.zeros(N)

        self.Nsquared = np.zeros(N)
        self.Msquared = np.zeros(N-1)
        self.pressGrad_x = np.zeros(len(self.y_mom))
        self.pressGrad_y = np.zeros(len(self.y_mom))        

        self.flag_brk = 0


        self.mode = mode

    def create_stretched_grid(self, no_zoom=10): #no_zoom number of zoomed cells
        dz_zoom = (self.y[1] - self.y[0]) / no_zoom
        self.y = np.insert(self.y, 1, np.linspace(self.y[0]+dz_zoom, self.y[1]-dz_zoom, no_zoom))

        dz_zoom = (self.y[-1] - self.y[-2]) / no_zoom
        self.y = np.insert(self.y, len(self.y)-1, np.linspace(self.y[-2]+dz_zoom, self.y[-1]-dz_zoom, no_zoom))

        dy = self.y[1:] - self.y[:-1]
        self.y_mom = self.y[:-1] + dy

        self.N = len(self.y)
        N = self.N
        self.k = np.zeros(N)
        self.eps = np.zeros(N)
        self.u = np.zeros(N-1)
        self.v = np.zeros(N-1)
        self.nu_t = np.zeros(N)
        self.nut_prime = np.zeros(N)
        #Beta phase variables, delete later
        self.Cmuvector = np.zeros(N)
        self.Cmuprimevector = np.zeros(N)
        self.alphaN = np.zeros(N)
        self.alphaM = np.zeros(N)
        self.norm_eps = np.ones(N)
        self.norm_u = np.ones(N-1)
        self.norm_v = np.ones(N-1)        
        self.norm_k = np.ones(N)
        self.rho_regridded = np.zeros(N)

        self.Nsquared = np.zeros(N)
        self.Msquared = np.zeros(N-1)
        self.pressGrad_x = np.zeros(len(self.y_mom))
        self.pressGrad_y = np.zeros(len(self.y_mom))
        

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
        self.u[:] = 0.0

    def guess_eps_values(self):
        self.eps[:] = 1e-8

    def guess_k_values(self):
        self.k[1:-1] = 1e-6 #1e-2*self.y[1:-1]

    def guess_diff_values(self):
        self.nu_t[1:-1] = 1e-6 #self.nu_t[0]
        self.nu_t[0] = 1e-3
        self.nu_t[-1] = 1e-3

    def set_BC(self):
        z0 = self.z0
        #surface ustar
        self.ustar[-1] = np.sqrt(self.rho_air/self.rho0 * self.C_Du ) * self.U10
        self.vstar[-1] = np.sqrt(self.rho_air/self.rho0 * self.C_Dv ) * self.V10
        #bottom ustar
        self.ustar[0] = self.init_u[0] * self.kappa / np.log( self.y_mom[0]/z0)
        self.vstar[0] = self.init_v[0] * self.kappa / np.log( self.y_mom[0]/z0)

        
        self.u[0] = self.ustar[0] * (5.5 + np.log(self.y[0] * abs(self.ustar[0])/self.nu)/self.kappa) #self.ORESVEL[0]
        self.u[-1] =  self.ustar[-1]*(5.5 + np.log(self.dy1_surf *abs(self.ustar[-1])/self.nu)/self.kappa) # self.ORESVEL[len(self.ORESVEL) - 1] #
        self.v[0] = self.vstar[0] * (5.5 + np.log(self.y[0] * abs(self.vstar[0])/self.nu)/self.kappa) #self.ORESVEL[0]
        self.v[-1] = self.vstar[0] * (5.5 + np.log(self.dy1_surf * abs(self.vstar[0])/self.nu)/self.kappa) #self.ORESVEL[0]
        ustar_resultant = np.sqrt(self.ustar**2 +self.vstar**2)
        self.k[0] = ustar_resultant[0]**2 / math.sqrt(self.Cmu)
        typ_eps = 1e-11 #typical eps in mid depth
        typ_nut = 1e-6 #typical nu_t set from measured top velocity
        self.k[self.N-1] = ustar_resultant[1]**2/math.sqrt(self.Cmu) #math.sqrt(typ_eps*typ_nut/self.Cmu) 
        
        self.eps[0] =  self.Cmu0**3 * self.k[0]**1.5 / (self.kappa * self.y[0]) #this variable is solved with a neumann BC, eps[0] is also computed, but this is the initial guess value
        self.eps[-1] = self.Cmu0**3 * self.k[-1]**1.5 / (self.kappa * self.dy1_surf) #self.u_star[1]**3/(self.kappa * self.dy1) #typ_eps ##
        
        self.nu_t[0] = self.kappa * abs(ustar_resultant[0]) * self.y[0] #self.Cmu*self.k[0]**2/self.eps[0]
        self.nu_t[-1] = self.kappa * abs(ustar_resultant[-1]) * self.dy1_surf #self.kappa * self.ustar[1] * self.dy1 #self.Cmu * self.k[self.N-1]**2/self.eps[self.N-1] #typ_nut #

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
        pdb.set_trace()
        self.Nsquared[1:] = -self.g/self.rho0 * (self.rho_regridded[1:] -
                                             self.rho_regridded[:-1]) / (self.y[1:] - self.y[:-1] )
        self.Nsquared[0] = -self.g/self.rho0 * (self.rho_regridded[1] -
                                           self.rho_regridded[0]) / (self.y[1] - self.y[0])

    def computeDiffusivities(self):
        self.nu_t[1:-1] = self.Cmuvector[1:-1] * self.k[1:-1]**2 / self.eps[1:-1] * self.Cmu0**3
        self.nu_t[0] = self.Cmu0**4 * self.k[0]**2 / self.eps[0]
        self.nu_t[-1] = self.Cmu0**4 * self.k[-1]**2 / self.eps[-1]
        
        self.nut_prime[1:-1] = self.Cmuprimevector[1:-1] * self.k[1:-1]**2/self.eps[1:-1] * self.Cmu0**3
        self.nut_prime[0] = self.Cmu0**4 * self.k[0]**2 / self.eps[0]
        self.nut_prime[-1] = self.Cmu0**4 * self.k[-1]**2 / self.eps[-1]

    def update_ustar(self):
        #bottom ustar
        self.ustar[0] = self.kappa / np.log(self.y_mom[0]/self.z0) * abs(self.u[0])
        self.vstar[0] = self.kappa / np.log(self.y_mom[0]/self.z0) * abs(self.v[0])
        ustar_resultant = np.sqrt(self.ustar[0]**2 + self.vstar[0]**2)
        self.z0 = 0.1*self.nu/ustar_resultant + 0.03*self.y_mom[0]
        if(self.z0 < 0.001):
            self.z0 = 0.001

    def bring_yplus_above50(self):
        pdb.set_trace()
        desired_yplus = 100
        ustar_resultant = np.sqrt(self.ustar[0]**2 + self.vstar[0]**2)
        new_y0 = desired_yplus * self.nu / ustar_resultant

        new_yt = np.linspace(new_y0, self.H-self.dy1_surf*1.1, self.N)
        self.dy = new_yt[1] - new_yt[0]
        new_ymom = new_yt[:-1]+self.dy*0.5
        #new_ymom[0] = new_yt[0] + self.dy1_bot*0.5
        #new_ymom[-1] = new_yt[-1] - self.dy1_surf*0.5
        pdb.set_trace()
        rho_interp_func = interpolate.interp1d(self.y, self.rho_regridded)
        self.rho_regridded = rho_interp_func(new_yt)
        
        press_grad_func = interpolate.interp1d(self.y_mom, self.pressGrad_x)
        self.pressGrad_x = press_grad_func(new_ymom)

        press_grad_func = interpolate.interp1d(self.y_mom, self.pressGrad_y)
        self.pressGrad_y = press_grad_func(new_ymom)

        u_interp_func = interpolate.interp1d(self.y_mom, self.u)
        self.u = u_interp_func(new_ymom)

        v_interp_func = interpolate.interp1d(self.y_mom, self.v)
        self.v = v_interp_func(new_ymom)

        k_interp_func = interpolate.interp1d(self.y, self.k)
        self.k = k_interp_func(new_yt)

        eps_interp_func = interpolate.interp1d(self.y, self.eps)
        self.eps = eps_interp_func(new_yt)

        nut_interp_func = interpolate.interp1d(self.y, self.nu_t)
        self.nu_t = nut_interp_func(new_yt)

        nut_interp_func = interpolate.interp1d(self.y, self.nut_prime)
        self.nut_prime = nut_interp_func(new_yt)


        self.y_mom = new_ymom
        self.y = new_yt
        
                        

    def compute_fully_coupled_2Neum_fImplicit(self):
        Nm = self.N-1
        Nt = self.N
        
        a = np.zeros((2*Nm+2*Nt , 2*Nm+2*Nt))
        b = np.zeros(2*Nm + 2*Nt)
        
        u_old = np.copy(self.u)
        v_old = np.copy(self.u)        

        dz = np.zeros(self.N-1)
        dz[:-1] = self.y_mom[1:] - self.y_mom[:-1]
        dz[-1] = self.y_mom[-1] - self.y_mom[-2]
        nu_avg = (self.nu_t[1:] + self.nu_t[:-1]) *0.5 + self.nu

        rho_avg = self.rho0*np.ones(len(b)) #(self.rho_regridded[1:] + self.rho_regridded[:-1])*0.5
        # u mom equation
        ########################################
        #####################################
        for i in range(0,Nm,1):
            if(i > 0): #coeff_{i-1}
                a[i, i-1] = -nu_avg[i-1] / dz[i-1]**2
            if(i < Nm-1): #coeff_{i+1}
                a[i, i+1] = -nu_avg[i+1] / dz[i+1]**2
            if(i > 0 & i<Nm-1): # coeff_{i}
                a[i, i] = 1./self.dt + 2 * nu_avg[i]/dz[i]**2
            if(i == 0): #coeff_i
                a[i, i] = 1./ self.dt + nu_avg[0]/dz[0]**2
                b[i] += -self.ustar[0]**2/dz[0]
            if(i == Nm-1): #coeff_i
                a[i, i] = 1./ self.dt + nu_avg[i]/dz[i]**2
                b[i] += self.ustar[-1]**2/dz[i]
            a[i, i+Nm] = -self.f
            b[i] += -self.pressGrad_x[i] / rho_avg[i] + u_old[i]/self.dt
            
        # v mom equation
        #####################################################
        #####################################################
        for i in range(Nm,2*Nm,1):
            j = i-Nm
            if(i > Nm): #coeff_{i-1}
                a[i, i-1] = -nu_avg[j-1] / dz[j-1]**2
            if(i < Nm-1): #coeff_{i+1}
                a[i, i+1] = -nu_avg[j+1] / dz[j+1]**2
            if(i > Nm & i < 2*Nm-1): # coeff_{i}
                a[i, i] = 1./self.dt + 2 * nu_avg[j]/dz[j]**2
            if(i == Nm): #coeff_i
                a[i, i] = 1./ self.dt + nu_avg[0]/dz[0]**2
                b[i] += -self.vstar[0]**2 / dz[0]
            if(i == 2*Nm-1): #coeff_i
                a[i, i] = 1./ self.dt + nu_avg[j]/dz[j]**2
                b[i] += self.vstar[-1]**2/dz[j]
            a[i, i-Nm] = self.f
            b[i] += -self.pressGrad_y[j] / rho_avg[j] + v_old[j]/self.dt

        #### k equation #########################
        ############################################
        #############################################
        nu_k = self.nu_t[:] / self.sigma_k
        k_old = np.copy(self.k)
        dz = np.zeros(self.N)
        dz[:-1] = abs(self.y[1:] - self.y[:-1])
        dz[-1] = abs(self.y[-1] - self.y[-2])

        buoy = np.zeros(Nt)
        prod = np.zeros(Nt)
        stable = (self.Nsquared[:] > 0.) # mask/filter to get bool for indices where stable is True
        unstable = np.invert(stable)
        buoy[stable] = -(self.nut_prime[stable]  * self.Nsquared[stable] / k_old[stable] )
        buoy[unstable] = - self.nut_prime[unstable] * self.Nsquared[unstable]
            
        msq_avg = np.zeros(Nt)
        msq_avg[1:-1] = (self.Msquared[:-1] + self.Msquared[1:]) * 0.5
        msq_avg[0] = self.Msquared[0]
        msq_avg[-1] = self.Msquared[-1]
        prod[:] = self.nu_t[:] * msq_avg[:]
        
        for i in range(2*Nm, 2*Nm+Nt, 1):
            j = i - 2*Nm
            if(j > 0):
                a[i, i-1] = -nu_k[j-1] / dz[j-1]**2
            if(j > 0 and j<Nt-1):
                a[i, i] = 2.* nu_k[j]/dz[j]**2 + 2. * self.eps[j]/k_old[j] + 1./self.dt
            if(j == 0):
                a[i, i] = 1.* nu_k[0]/dz[0]**2 + 2. * self.eps[0]/k_old[0] + 1./self.dt
            if(j == Nt-1):
                a[i, i] = 1.* nu_k[-1]/dz[-1]**2 + 2. * self.eps[-1]/k_old[-1] + 1./self.dt
            if(stable[j] == True):
                a[i, i] += -buoy[j]
            if(j < Nt-1):
                a[i, i+1] = -nu_k[j+1]/dz[j+1]**2
                
            b[i] = prod[j] + self.eps[j] + k_old[j] / self.dt
            if(unstable[j]):
                b[i] += buoy[j]

        ## eps eps eps eps eps eps eps eps eps eps
        #######################################################
        ####################################################
        nu_eps = self.nu_t[:]/self.sigma_eps
        
        Ceps1 = 1.44
        Ceps2 = 1.92
        Ceps3 = np.zeros(len(self.Nsquared))
        Cmu0 = 0.5562
        
        oldeps = np.copy(self.eps)
        Ceps3[(self.Nsquared < 0.)] = 1. #if unstable strat
        Ceps3[(self.Nsquared > 0.)] = -0.4 #if stable strat

        buoy[:] = 0.0
        prod[:] = 0.0
            
        pat = (Ceps3 * self.Nsquared > 0) # if the sink term is positive, do patankar type discretization
        nopat = np.invert(pat) # else, leave sink term on the rhs, ie. as is
        buoy[pat] = Ceps3[pat] * self.Cmuprimevector[pat]  * self.Nsquared[pat] * self.k[pat]/oldeps[pat] # * self.Cmu0**3
        buoy[nopat] = -Ceps3[nopat] * self.Cmuprimevector[nopat] * self.k[nopat] * self.Nsquared[nopat] # *self.Cmu0**3
        
        if(pat[0] == True):
            buoy[0] = Ceps3[0] * self.Cmu0**4  * self.Nsquared[0] * self.k[0]/oldeps[0] # * self.Cmu0**3
        else:
            buoy[0] = -Ceps3[0] * self.Cmu0**4 * self.k[0] * self.Nsquared[0] # *self.Cmu0**3
        if(pat[-1] == True):
            buoy[-1] = Ceps3[-1] * self.Cmu0**4  * self.Nsquared[-1] * self.k[-1]/oldeps[-1] # * self.Cmu0**3 
        
        coeff_nl_lhs = Ceps2 * 2.* oldeps / self.k
        coeff_nl_rhs = Ceps2 * oldeps**2 / self.k
        
        prod[1:-1] = Ceps1 * self.Cmuvector[1:-1] * self.k[1:-1] * msq_avg[1:-1] # * Cmu0**3
        prod[0] = Ceps1 * self.Cmu0**4 * self.k[0] * msq_avg[0] # * Cmu0**3
        prod[-1] = Ceps1 * self.Cmu0**4 * self.k[-1] * msq_avg[-1] # * Cmu0**3        
        
        for i in range(2*Nm+Nt, 2*Nm+2*Nt, 1):
            j = i - (2*Nm+Nt)

            if(j>0 and j<Nt-1):
                a[i,i] = 2. * nu_eps[j] / dz[j]**2 + coeff_nl_lhs[j] + 1/self.dt
            if(j==0):
                a[i,i] = nu_eps[0] / dz[0]**2 + coeff_nl_lhs[0] + 1/self.dt
                rfict0 = nu_eps[0]/dz[0] * self.k[0]**(1.5)/(self.kappa * self.y[0]**2) * self.Cmu0**3
                b[i] += rfict0
            if(j==Nt-1):
                a[i,i] = nu_eps[j] / dz[j]**2 + coeff_nl_lhs[j] + 1/self.dt
                rfictN = -nu_eps[j] / dz[j] * self.k[j]**(1.5)/(self.kappa * self.dy1_surf**2) * self.Cmu0**3
                b[i] += rfictN
            if(pat[j]):
                a[i,i] += buoy[j]
            if(j > 0):
                a[i,i-1] = -nu_eps[j-1] / dz[j-1]**2
            if(j < Nt-1):
                a[i,i+1] = -nu_eps[j+1] / dz[j+1]**2

            b[i] = prod[j] + coeff_nl_rhs[j] + oldeps[j]/self.dt
            if(nopat[j]):
                b[i] += buoy[j]
                
        a = csr_matrix(a)
        #b = csr_matrix(b)
        
        uvke = spsolve(a,b)
        self.u[:] = uvke[0:Nm]
        self.v[:] = uvke[Nm:2*Nm]
        self.k[:] = uvke[2*Nm: 2*Nm+Nt]
        self.k[np.where(self.k < 3e-8)] = 3e-8 # k limiter        
        self.eps[:] = uvke[2*Nm+Nt : 2*Nm+2*Nt]
        mask1 = self.Nsquared > 0
        limiter = 0.045 * self.k**2 * self.Nsquared
        mask2 = self.eps**2 < limiter
        #pdb.set_trace()        
        self.eps[mask1 & mask2] = np.sqrt(limiter[mask1 & mask2]) # eps limiter
        
        self.norm_u[:] = abs(u_old[:] - self.u[:])/abs(self.u[:])
        self.norm_v[:] = abs(v_old[:] - self.v[:])/abs(self.v[:])
        self.norm_k[:] = abs(k_old[:] - self.k[:])/abs(self.k[:])
        self.norm_eps[:] = abs(self.eps[:] - oldeps[:])/abs(self.eps[:].max())        
    

    def solver(self, MAXITER, DT, SAVE_INTERVAL, tol=1e-3):
        self.tol = tol
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
            self.computeStabilityFunction()
            self.computeMsquared()            
            self.compute_fully_coupled_2Neum_fImplicit()
            self.computeDiffusivities()
            self.update_ustar()
            self.iter += 1
            ustar_resultant = np.sqrt(self.ustar**2 + self.vstar**2)
            ymom_plus_bottom = self.y_mom[0] * ustar_resultant[0] / self.nu #((self.nu_t[0] + self.nu_t[1])*0.5)
            ymom_plus_surf = (self.H - self.y_mom[-1]) * ustar_resultant[-1] / self.nu #((self.nu_t[-1] + self.nu_t[-2])*0.5)
            yt_plus_bottom = self.y[0] * ustar_resultant[0] / self.nu #((self.nu_t[0] + self.nu_t[1])*0.5)
            if(self.mode == "verbose"):
                print("iter = "+str(self.iter)+", zo_bt = "+str(self.z0)+", y+_bt_{t,mom},mom_surf = "+str(yt_plus_bottom)+", "+str(ymom_plus_bottom)+", "+str(ymom_plus_surf))
                # ", max(norm_k)= "+str(max(self.norm_k))+", norm_eps_max="+str(max(self.norm_eps[:-1]))+", norm_u_max="+str(max(self.norm_u))+", norm_v_max="+str(max(self.norm_v))+
            if(self.iter>100 and yt_plus_bottom <= 50):
                self.bring_yplus_above50()
                
            if(save_int != 0):
                if(self.iter % save_int == 0):
                    print("Case: "+str(self.case))
                    self.WriteOutput()
                    self.plotter("single")
                    
            if(self.iter > 1000 and (yt_plus_bottom > 1e3 or ymom_plus_bottom > 1e3 or ymom_plus_bottom < 20)):
                print("y_mom[0] has moved out of the logarithmic bottom Boundary layer!!! Stopping simulation!!!!")
                break

            ## if(max(self.norm_u) < self.tol and max(self.norm_v)<self.tol and max(self.norm_k)<self.tol and max(self.norm_eps)<self.tol):
            ##     break
            
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
        ax.set_xlim(-1,1)
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
