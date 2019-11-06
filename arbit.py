def compute_OHC(CTEMP, mask, h_b=-500, lon_bins = np.arange(0, 361, 5), rho0=1027.7, Cp=3850, SA_max=34.85 ):
    select_Z = CTEMP.Z >= h_b

    Tf0 = gsw.t_freezing(SA_max, 0 , 0) # SA_max set to 34.85 from the analysis of field observations (MEOP, Argo, SOCCOM)
    
    dtheta = (CTEMP.CT.where(mask).sel(Z = CTEMP.Z[select_Z]).mean('Time').groupby_bins('XC', lon_bins ).mean('XC').groupby('XC_bins').mean('YC') - Tf0).compute()

    rho0 = 1027.7
    Cp = 3850
    
    OHC = (rho0 * Cp * dtheta).compute()    
