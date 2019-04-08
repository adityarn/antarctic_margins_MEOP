import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw
import pdb
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image
import matplotlib.colors as colors
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point

import plot_stations as pltStn
import importlib
importlib.reload(pltStn)

dfmg = pd.read_csv("dfmg_glDist.csv")
del dfmg['Unnamed: 0']
del dfmg['Unnamed: 0.1']
dfmg.loc[:,'JULD'] = pd.to_datetime(dfmg.loc[:, 'JULD'])

WSO_source = (dfmg['CTEMP'] <= -1.7 ) & (dfmg["PSAL_ADJUSTED"] > 34.5)
WSO_prod = (dfmg['CTEMP'] <= -0.8) & (dfmg['CTEMP'] >= -1.2) & (dfmg["PSAL_ADJUSTED"] > 34.5)
def get_mask_from_prof_mask(df, profmask):
    profs = dfmg.loc[profmask, 'PROFILE_NUMBER'].unique()
    mask = dfmg.loc[:, 'PROFILE_NUMBER'].isin(profs)
    
    return mask

WSO_source = get_mask_from_prof_mask(dfmg, WSO_source)
WSO_prod = get_mask_from_prof_mask(dfmg, WSO_prod)

box1 = (dfmg["LATITUDE"] > -80) & (dfmg["LATITUDE"] < -75) & (dfmg["LONGITUDE"] > -60) & (dfmg["LONGITUDE"] < -40)
box2 = (dfmg["LATITUDE"] > -80) & (dfmg["LATITUDE"] < -75) & (dfmg["LONGITUDE"] > -40) & (dfmg["LONGITUDE"] < -20)
box3 = (dfmg["LATITUDE"] > -75) & (dfmg["LATITUDE"] < -70) & (dfmg["LONGITUDE"] > -60) & (dfmg["LONGITUDE"] < -40)
box4 = (dfmg["LATITUDE"] > -75) & (dfmg["LATITUDE"] < -70) & (dfmg["LONGITUDE"] > -40) & (dfmg["LONGITUDE"] < -20)

box5 = (dfmg["LATITUDE"] > -80) & (dfmg["LATITUDE"] < -75) & (dfmg["LONGITUDE"] > 160) & (dfmg["LONGITUDE"] < 180)
box6 = (dfmg["LATITUDE"] > -80) & (dfmg["LATITUDE"] < -75) & (dfmg["LONGITUDE"] > -180) & (dfmg["LONGITUDE"] < -160)
box7 = (dfmg["LATITUDE"] > -75) & (dfmg["LATITUDE"] < -70) & (dfmg["LONGITUDE"] > 160) & (dfmg["LONGITUDE"] < 180)
box8 = (dfmg["LATITUDE"] > -75) & (dfmg["LATITUDE"] < -70) & (dfmg["LONGITUDE"] > -180) & (dfmg["LONGITUDE"] < -160)

box9 = (dfmg["LATITUDE"] > -70) & (dfmg["LATITUDE"] < -65) & (dfmg["LONGITUDE"] > 60) & (dfmg["LONGITUDE"] < 70)
box10 = (dfmg["LATITUDE"] > -70) & (dfmg["LATITUDE"] < -65) & (dfmg["LONGITUDE"] > 70) & (dfmg["LONGITUDE"] < 80)

Weddell = (box1 | box2 | box3| box4)
Ross = (box5 | box6 | box7 | box8)
Prydz = (box9 | box10)
EBS = (dfmg['LONGITUDE'] > -80) & (dfmg['LONGITUDE'] < -60) #Eastern Bellingshausen Sea
WBS = (dfmg['LONGITUDE'] > -100) & (dfmg['LONGITUDE'] < -80) # Western Bellingshausen Sea

year_mask = dfmg.loc[:, 'JULD'].dt.year == 2011
month_mask = dfmg.loc[:, 'JULD'].dt.month == 3

mask_selection = year_mask & month_mask & Weddell & WSO_source
time_vector = dfmg.loc[dfmg.loc[mask_selection].\
                       groupby('PROFILE_NUMBER').head(1).index, 'JULD'].astype(pd.datetime).values

profile_numbers = dfmg.loc[dfmg.loc[mask_selection].\
                           groupby('PROFILE_NUMBER').head(1).index, 'PROFILE_NUMBER'].values

sorted_indices = np.argsort(time_vector)
time_vector = time_vector[sorted_indices]
profile_numbers = profile_numbers[sorted_indices]
diff = np.diff(time_vector)
for i in range(len(diff)):
    diff[i] = diff[i].total_seconds()/3600.
len(diff)




#profs = profile_numbers[np.where(diff <= 1)[0]][:]
# 1st set of points: np.array([257898, 259575, 256589, 258595, 259075, 261291, 259076, 256590, 261292, 258596, 257899])
profs = np.array([259580, 261295, 259081, 260784, 256594, 257904, 260219, 258599,
       256595, 261296, 259082, 256573, 256574]) 
mask = dfmg.loc[:, 'PROFILE_NUMBER'].isin(profs)
positions = dfmg.loc[dfmg[mask].groupby('PROFILE_NUMBER').tail(1).index, 'LATITUDE':'LONGITUDE'].values
markers = dfmg.loc[dfmg[mask].groupby('PROFILE_NUMBER').tail(1).index, 'PROFILE_NUMBER'].values
pltStn.plot_station_locations(positions, title='Profile Locations', wd=7, ht=7, 
                              region='Weddell', markers=markers)
