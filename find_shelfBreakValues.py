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
import mplcursors
import cartopy
import fiona
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.ticker as ticker


def findShelfBreakValues(acASF, dfmg, MEOPDIR = "/media/data"):
    profs = acASF.PROFILE_NUMBERS
    profs = [int(x) for x in profs.split(',')]
    year = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[0]]), "JULD"].dt.year.values[0]
    month = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[0]]), "JULD"].dt.month.values[0]
    region = acASF.REGION


    ctemps = []
    latlons = []
    depth = []
    gamman = []
    CTEMP = []
    echodepth = []
    dist_p2p_all = [] #chainage distance of each data point
    dist_p2p = []     # chainage distance of each profile location
    sb_ind = acASF.SHELF_BREAK_ind
    #sb_longitude = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[sb_ind]]), 'LONGITUDE'].values[0]
    sb_gn = np.array([])
    no_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    sb_longitude = np.nan
    try:
        for i in range(len(profs)):
            dfSelect = dfmg.PROFILE_NUMBER.isin([profs[i]])
            ctemps = np.concatenate((ctemps, dfmg.loc[dfSelect, "CTEMP"].values))
            latlons.append([dfmg.loc[dfSelect, 'LATITUDE'].values[0], dfmg.loc[dfSelect, 'LONGITUDE'].values[0] ])
            depth = np.concatenate((depth,dfmg.loc[dfSelect, "DEPTH"].values))
            gamman = np.concatenate((gamman, dfmg.loc[dfSelect, "gamman"].values))
            CTEMP = np.concatenate((CTEMP, dfmg.loc[dfSelect, "CTEMP"].values))
            echodepth = np.concatenate((echodepth, [dfmg.loc[dfSelect].ECHODEPTH.values[0]]))

            if(i == 0):
                dist_p2p.append(0)
                dist_p2p_all = np.concatenate((dist_p2p_all, np.zeros(len(dfmg[dfSelect])) ))
            else:
                dist_p2p.append(dist_p2p_all[-1] + haversine(latlons[i-1], latlons[i]))
                dist_p2p_all = np.concatenate( (dist_p2p_all, np.zeros(len(dfmg[dfSelect])) + dist_p2p[-1]))

        latlons = np.array(latlons)

        dist_echo = np.unique(dist_p2p_all)
        ndist = int(np.max(dist_p2p_all) / 2.)

        dist_grid = np.linspace(np.min(dist_p2p_all), np.max(dist_p2p_all), ndist)
        ndepth = int(-np.min(depth) / 10.)
        depth_grid = np.linspace(np.min(depth), 0, ndepth)

        dist_grid, depth_grid = np.meshgrid(dist_grid, depth_grid)
        gamman_interpolated = griddata(np.array([dist_p2p_all, depth]).T, gamman, (dist_grid, depth_grid), method='cubic' )
        CTEMP_interpolated =  griddata(np.array([dist_p2p_all, depth]).T, CTEMP, (dist_grid, depth_grid), method='cubic' )

        try:
            echodepth_interpolater = interp1d(dist_p2p, echodepth, kind='cubic')
            dist_interpolated = np.linspace(np.min(dist_p2p_all), np.max(dist_p2p_all), ndist)
            echodepth_interpolated = echodepth_interpolater(dist_interpolated)
        except:
            echodepth_interpolater = interp1d(dist_p2p, echodepth, kind='linear')
            dist_interpolated = np.linspace(np.min(dist_p2p_all), np.max(dist_p2p_all), ndist)
            echodepth_interpolated = echodepth_interpolater(dist_interpolated)

        try:
            slope_break_location = dist_interpolated[(echodepth_interpolated < -800) & (echodepth_interpolated > -1200)][0]
        except:
            try:
                slope_break_location = dist_interpolated[(echodepth_interpolated < -1500)][0]
            except:
                slope_break_location = dist_interpolated[(echodepth_interpolated < -2500)][0]

        depth_interpolated = np.linspace(np.min(depth), 0, ndepth)
        sb_gridded_index = np.where(dist_interpolated == slope_break_location)[0][0]

        sb_gn = gamman_interpolated[:, sb_gridded_index]
        sb_CT = CTEMP_interpolated[:, sb_gridded_index]
        if ~np.isnan(sb_ind):
            sb_longitude = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[int(sb_ind)]]), 'LONGITUDE'].values[0]
        else:
            sb_longitude = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[-1]]), 'LONGITUDE'].values[0]

    except:
        pass
   
    if sb_gn.any():
        try:
            depth_28gn = depth_interpolated[(sb_gn >= 27.95) & (sb_gn <= 28.05)][-1]
        except:
            depth_28gn = np.nan
        try:
            depth_281gn = depth_interpolated[(sb_gn > 28.05) & (sb_gn <= 28.15)][-1]
        except:
            depth_281gn = np.nan
        try:
            depth_2827gn = depth_interpolated[(sb_gn >= 28.22) & (sb_gn < 28.27)][-1]
        except:
            depth_2827gn = np.nan
        try:
            depth_gt2827gn = depth_interpolated[sb_gn > 28.27][-1]
        except:
            depth_gt2827gn = np.nan

        try:
            CT_28gn = sb_CT[(sb_gn >= 27.95) & (sb_gn <= 28.05)][-1]
        except:
            CT_28gn = np.nan
        try:
            CT_281gn = sb_CT[(sb_gn >= 28.05) & (sb_gn < 28.15)][-1]
        except:
            CT_281gn = np.nan
        try:
            CT_2827gn = sb_CT[(sb_gn >= 28.22) & (sb_gn < 28.27)][-1]
        except:
            CT_2827gn = np.nan
        try:
            CT_gt2827gn = sb_CT[sb_gn > 28.27][-1]
        except:
            CT_gt2827gn = np.nan
        print(sb_longitude, slope_break_location, depth_28gn, depth_2827gn, depth_gt2827gn)
    else:
        depth_28gn = np.nan
        depth_281gn = np.nan
        depth_2827gn = np.nan
        depth_gt2827gn = np.nan
        
        CT_28gn = np.nan
        CT_281gn = np.nan
        CT_2827gn = np.nan
        CT_gt2827gn = np.nan

    return {'depth_28gn': depth_28gn, 'depth_2827gn':depth_2827gn, 'depth_gt2827gn': depth_gt2827gn, 'CT_28gn': CT_28gn, 'CT_2827gn': CT_2827gn, 'CT_2827gn': CT_2827gn, 'CT_gt2827gn': CT_gt2827gn,
            'depth_281gn': depth_281gn, 'CT_281gn': CT_281gn,
            'SHELF_BREAK_LONGITUDE':sb_longitude}
