import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors
from haversine import haversine
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import matplotlib
import xarray as xr
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar
import importlib
import pdb
import plot_topView_contourf as topView
importlib.reload(topView)

def compute_bathymetryGradients(lonstep=5, latstep=1):
    bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')

    lonlen = len(bathy.lon)
    lonindices = np.arange(0, lonlen+1, lonstep)
    lonindices[-1] = lonindices[-1] - 1
    bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, latstep))

    ## dx = float(abs(bathyS.lon[1] - bathyS.lon[0]))
    ## dy = float(abs(bathyS.lat[1] - bathyS.lat[0]))
    ## bathy_x, bathy_y = np.gradient(bathyS.elevation.where(bathyS.elevation <= 0).values, dx, dy)
    r = 6371e3
    bathy_x = np.zeros_like(bathyS.elevation.values)
    bathy_y = np.zeros_like(bathyS.elevation.values)
    delta_lon = float(abs(bathyS.lon[1] - bathyS.lon[0]))
    
    for li in range(len(bathyS.lat)):
        if(li < len(bathyS.lat)-1):
            delta_lat = float(abs(bathyS.lat[li] - bathyS.lat[li+1]))
        else:
            delta_lat = float(abs(bathyS.lat[li] - bathyS.lat[li-1]))
            
        dy = float(r * np.deg2rad(delta_lat))
        dx = abs(float(r * np.sin(np.deg2rad(bathyS.lat[li]) ) * delta_lon))
        if(li > 0):
            bathy_y[li, :] = (bathyS.elevation[li, :] - bathyS.elevation[li-1, :]) / dy
        else:
            bathy_y[li, :] = (bathyS.elevation[li+1, :] - bathyS.elevation[li, :]) / dy

        bathy_x[li, 1:] = (bathyS.elevation.values[li, 1:] - bathyS.elevation.values[li, :-1]) / dx
        bathy_x[li, 0] = bathy_x[li, 1]

            
    return np.sqrt(bathy_x**2 + bathy_y**2)

    

def plot_bathymetryGradients(m=None, bathy_gradient=None, bathyS=None, plotBathy=True, bathy=None):
    if not bathy_gradient:
        bathy_gradient = compute_bathymetryGradients()

    if not bathyS:
        bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')

        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 5)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 1))
        
    ## dx = float(abs(bathyS.lon[1] - bathyS.lon[0]))
    ## dy = float(abs(bathyS.lat[1] - bathyS.lat[0]))
    ## bathy_x, bathy_y = np.gradient(bathyS.elevation.where(bathyS.elevation <= 0).values, dx, dy)
    ## bathy_gradient = ma.masked_array(np.sqrt(bathy_x**2 + bathy_y**2)    )
    bathy_gradient = np.abs(ma.masked_array(np.array(bathy_gradient) ))
    bathy_gradient.mask = (bathy_gradient < 0.002) | (bathyS.elevation < -2500) | (bathyS.elevation > -800)

    if not m:
        m = topView.createMapProjections(-90, 0, region="Whole")

    topView.plot_scalar_field(bathy_gradient, bathyS.lon, bathyS.lat, m=m, levs=list(np.linspace(0,0.05,20)), meridians=list(np.arange(-180, 180, 10)), bathy=bathy )
