import xarray as xr
import pandas as pd
import numpy as np

# PE is the merged and climatological xarray dataset of ERA_interim mdfa evap and precip fields
def compute_precip_less_evap(PE, bathy, latmin, latmax, lonmin, lonmax):
    if(lonmin < 0):
        lonmin = lonmin + 360
    if(latmin < -):
        latmin = latmin + 360
    PElats = PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).latitude
    PElons = PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).longitude
    
    precip_total = PE.tp.sel(longitude=slice(lonmin, lonmax), latitude=slice(latmax, latmin)).sum()
    evap_total = PE.e.sel(longitude=slice(lonmin, lonmax), latitude=slice(latmax, latmin)).sum()

    return [precip_total, evap_total]

def sea_ice_flux(seaIce, latmin, latmax, lonmin, lonmax, timeStart):
