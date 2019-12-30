import xarray as xr
import pandas as pd
import gsw
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import sys
import importlib
import matplotlib.gridspec as gridspec
#from pandarallel import pandarallel
import pdb
import multiprocessing as mp

#Freezing temperature selected for the 99th percentile surface SA value of 34.85 g/kg
def compute_lonbinned_OHC(gdf, dz, h_b, rho0, Cp, Tf0):
    try:
        # MEOP data is compressed using broken stick method. Here, we interpolate them onto a 2m depth interval to match the Argo and SOCCOM data
        dfsel_MEOP = gdf.DATASET == "MEOP"
        #p = mp.Pool(mp.cpu_count()) # Data parallelism Object
        meop_interpolated =  np.concatenate(gdf[dfsel_MEOP].groupby("PROFILE_NUMBER").apply(interpolate_MEOP).values, axis=0)
        #meop_interpolated = p.map(interpolate_MEOP, gdf[dfsel_MEOP].groupby("PROFILE_NUMBER"))

        meop_pot_temp = gsw.pt_from_t(meop_interpolated[:, 0], meop_interpolated[:, 1] , meop_interpolated[:, 2] , 0.0 )

        dfsel_argo_soccom = (gdf.DATASET == "Argo") | (gdf.DATASET == "SOCCOM")
        total_length = len(meop_pot_temp) + len(gdf[dfsel_argo_soccom])
        df_concat = pd.DataFrame({'DEPTH': pd.Series(np.concatenate((meop_interpolated[:, 3], gdf[dfsel_argo_soccom].DEPTH.values), axis=0 ), index=np.arange(0, total_length) ),
                                  'POT_TEMPERATURE': pd.Series(np.concatenate( (meop_pot_temp, gdf[dfsel_argo_soccom].POT_TEMPERATURE.values ), axis=0), index=np.arange(0, total_length) ) } )


        #df_concat = pd.concat([meop_interpolated, gdf[dfsel_argo_soccom] ], ignore_index = True, sort=False)

        pot_temp_depth_binned_mean = df_concat.POT_TEMPERATURE.groupby(pd.cut(df_concat.DEPTH, np.arange(h_b, 0.1, dz))).mean()
        pot_temp_depth_binned_std = df_concat.POT_TEMPERATURE.groupby(pd.cut(df_concat.DEPTH, np.arange(h_b, 0.1, dz))).std()    

        OHC = np.sum(rho0 * Cp * (pot_temp_depth_binned_mean - Tf0) * dz)

        a = rho0 * Cp * dz
        OHC_sigma = np.sqrt(np.sum(a**2 * pot_temp_depth_binned_std**2))

        return np.array([OHC, OHC_sigma])
    except:
        return np.array([np.nan, np.nan])
    

def compute_OHC(df, ax, h_b = -500, dz=5, lon_bins = np.arange(0, 360.01, 5), ymin=0, ymax=6e9, hide_yticks=False ):
    #pandarallel.initialize()
    rho0 = 1027.7 #kg/m3
    Cp = 3850 #J/kg/K heat capacity of sea water
    Tf0 = gsw.t_freezing(34.85, 0 , 0)
    #dfsel = (df.DATASET == "Argo") | (df.DATASET == "SOCCOM") | (df.DATASET == "MEOP")
    
    OHC = np.stack(df.groupby(pd.cut(df.LONGITUDE, lon_bins) ).\
                                                          apply(compute_lonbinned_OHC, dz, h_b, rho0, Cp, Tf0).values) #* 1e-9
    OHC[ (OHC[:, 0] == 0) , 0] = np.nan
    #OHC column 0 has the heat content vector in longitudinal bins.
    #OHC column 1 has the standard deviation of the heat content vector in each longitudinal bin.
    lon_bins = (lon_bins[1:] + lon_bins[:-1])*0.5
    #lon_bins[lon_bins < 0] = lon_bins[lon_bins < 0] + 360
    
    ax.fill_between(np.sort(lon_bins), (OHC[:, 0]-OHC[:, 1])[np.argsort(lon_bins)], (OHC[:, 0]+OHC[:, 1])[np.argsort(lon_bins)], facecolor="coral", edgecolor="r", alpha=0.5)
    #ax.errorbar(lon_bins[1:], OHC[:, 0], yerr = OHC[:, 1], fmt=".", color="r", markersize=3, capsize=3)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_xticks(np.arange(30, 361, 60), minor=True)
    ax.set_xticklabels(np.arange(0, 361, 60), rotation=90)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 360)
    if hide_yticks:
        ax.set_yticklabels([])
    ax.grid()    
    
    return OHC
    



def interpolate_MEOP(gdf):
    try:
        depth_top = gdf.DEPTH.values[0]
        depth_bottom = gdf.DEPTH.values[-1]
        depth = gdf.DEPTH.values
        original_index = gdf.index
        ind = 0
        for i in range(len(depth)-1):
            bins = (depth[ind] - depth[ind+1]) // 2.0
            depth_interpd = np.linspace(depth[ind+1], depth[ind], bins)[::-1]
            depth = np.insert(depth, ind+1, depth_interpd[1:-1])
            ind = ind + len(depth_interpd[1:-1]) + 1

        depth_reindexed = depth

        df2 = gdf
        df2 = df2.set_index(df2.DEPTH)

        df2 = df2.reindex(depth_reindexed, method=None)
        df2 = df2.interpolate(method='linear')
        df2['SA'] = df2.SA.interpolate(method='linear')
        df2['TEMP_ADJUSTED'] = df2.TEMP_ADJUSTED.interpolate(method='linear')
        df2['PRES_ADJUSTED'] = df2.PRES_ADJUSTED.interpolate(method='linear')

        # df2['JULD'] = df2.JULD.interpolate(method='pad')
        # df2['PLATFORM_NUMBER'] = df2.PLATFORM_NUMBER.interpolate(method='pad')
        # df2['POSITION_QC'] = df2.POSITION_QC.interpolate(method='pad')
        # df2['PSAL_ADJUSTED_QC'] = df2.PSAL_ADJUSTED_QC.interpolate(method='pad')
        # df2['SHELF_BREAK_PROFILE'] = df2.SHELF_BREAK_PROFILE.interpolate(method='pad')
        # df2['SHELF_PROFILE'] = df2.SHELF_PROFILE.interpolate(method='pad')

        # del(df2['DEPTH'])
        # df2.reindex(original_index)
        #print(np.array([df2.SA.values, df2.TEMP_ADJUSTED.values, df2.PRES_ADJUSTED.values]).shape)
        return np.array([df2.SA.values, df2.TEMP_ADJUSTED.values, df2.PRES_ADJUSTED.values, depth]).T
    except:
        return np.array([ [np.nan], [np.nan], [np.nan], [np.nan]]).T
