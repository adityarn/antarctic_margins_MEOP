def integrate_vertically(gdf):
    return (gdf * np.abs(np.diff(CTEMP.Z[0:60]))).sum()

def compute_OHC(dtheta, ax, h_b=-500, lon_bins = np.arange(0, 361, 5), rho0=1027.7, Cp=3850, SA_max=34.85 ):
    
    dtheta = (CTEMP.CT.where(mask).sel(Z = CTEMP.Z[select_Z]).mean('Time').groupby_bins('XC', lon_bins ).mean('XC').groupby('XC_bins').mean('YC') - Tf0).compute()
    dtheta_sigma = (CTEMP.CT.where(mask).sel(Z = CTEMP.Z[select_Z]).std('Time').groupby_bins('XC', lon_bins ).mean('XC').groupby('XC_bins').mean('YC') - Tf0).compute()

    rho0 = 1027.7
    Cp = 3850
    
    OHC = (rho0 * Cp * dtheta).compute()

    OHC = OHC.rename({'depth':'OHC'})
    
    OHC.OHC.groupby('XC_bins').apply(integrate_vertically).plot()




