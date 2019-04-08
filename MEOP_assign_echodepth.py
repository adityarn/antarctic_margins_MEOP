
# # coding: utf-8

# # # MEOP dataset analysis
# # MEOP data can be found at: http://www.meop.net/database/format.html
# # MEOP stands for: "Marine Mammals Exploring the Oceans from Pole to Pole".
# # MEOP consists of ARGO like profiles, however, the instruments are fitted onto marine mammals.
# # The UK dataset of MEOP contains many profiles in the Weddel Sea region, the following analysis is of this dataset.

# # In[29]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw
from netCDF4 import Dataset
import xarray as xr
from scipy.interpolate import griddata
from scipy.io import netcdf

dfm = pd.read_csv("ncARGOmergedpandas.csv")
dfm["JULD"] = pd.to_datetime(dfm['JULD'],infer_datetime_format=True)
below60S = dfm["LATITUDE"] < -60
mask_notbad_temp = ~(dfm['TEMP_ADJUSTED_QC'] == 4)
mask_notbad_sal = ~(dfm['PSAL_ADJUSTED_QC'] == 4)
mask_notbad_pres = ~(dfm['PRES_ADJUSTED_QC'] == 4)

dfmg = dfm[below60S & mask_notbad_pres & mask_notbad_sal & mask_notbad_temp] # data with only good QC + null value flags

bathy = netcdf.netcdf_file('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc', 'r')

lons = bathy.variables['lon'][:]
lats = bathy.variables['lat'][:]
lonbounds = [-180, 180] # creating lon,lat bounds for region of interest, ie. Weddel Sea
latbounds = [-80, -60]

latli = np.argmin( np.abs(lats - latbounds[0]) ) # extracting the indices of start and end of the region of interest
latui = np.argmin( np.abs(lats - latbounds[1]) )
lonli = np.argmin( np.abs(lons - lonbounds[0]) )
lonui = np.argmin( np.abs(lons - lonbounds[1]) )
elev_below60S = bathy.variables['elevation'][latli:latui, lonli:lonui] # elevation of SO below 60S
mask_below0 = elev_below60S[:,:] > 0
elev_below60S = np.ma.masked_array(elev_below60S[:,:] , mask_below0)

unique_positions = dfmg.loc[:,"LATITUDE": "LONGITUDE"].drop_duplicates(subset=['LONGITUDE', 
                                                                               'LATITUDE']).values

dfm['ECHODEPTH'] = np.nan

print("Beginning to assign echodepth")

def assign_echodepth(df):
    ind = df.index
    lon = df.loc[ind[0], 'LONGITUDE']
    lat = df.loc[ind[0],'LATITUDE']
    ind_lon = np.argmin(np.abs(lons - lon))
    ind_lat = np.argmin(np.abs(lats - lat))
    df.loc[:,'ECHODEPTH'] = bathy.variables['elevation'][ind_lat, ind_lon]

    return df
    
dfmg = dfmg.groupby(['LATITUDE', 'LONGITUDE']).apply(assign_echodepth)

dfm.loc[dfmg.index] = dfmg

dfmg.to_csv("dfmg.csv")
