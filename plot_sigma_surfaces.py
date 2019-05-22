import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw
import importlib
import numpy.ma as ma
import cartopy.crs as ccrs
import pdb
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image
import matplotlib.colors as colors
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import gaussian_kde
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec 
from matplotlib.colorbar import Colorbar 
import sys

def find_topOfSigmaSurfaceDepth(gdf):
    return gdf.loc[gdf.groupby("PROFILE_NUMBER").head(1).index].DEPTH.mean()

def find_topOfSigmaSurfaceDepth_corr_CTEMP(gdf):
    return gdf.loc[gdf.groupby("PROFILE_NUMBER").head(1).index].CTEMP.mean()
def find_topOfSigmaSurfaceDepth_corr_sal(gdf):
    return gdf.loc[gdf.groupby("PROFILE_NUMBER").head(1).index].PSAL_ADJUSTED.mean()


def plot_slope_sigma0_surfaces_CF(dfmg, sigma_surfaces=[27.67, 27.74, 27.86], surface_type=["equal_to", "equal_to", 
                                        "greater_than"], save=False, savename="Untitled.png", tol=0.01, 
                              wd=190/25.4, ht=230/25.4):
    dfsel = (dfmg.loc[:, 'LONGITUDE'] < 0)
    dfmg.loc[dfsel, 'LONGITUDE'] = dfmg.loc[dfsel, 'LONGITUDE'] + 360
    plt.close(1)
    plt.figure(1, figsize=(wd, ht))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1,1,1,0.01, 0.1], wspace=0.01, hspace=0.01)
    
    vmin=[-1500, -1.5, 34.4]
    vmax = [0.1, 1.5, 34.8]
    extend = ['min', 'both']
    levs = [np.arange(vmin[0], vmax[0], 150), np.arange(vmin[1], vmax[1]+0.01, 0.05), np.arange(vmin[2], vmax[2]+0.01, 0.005)]
    cmap = ['viridis', 'bwr', 'viridis']
    CF = []
    ax = []
    subplot_titles = ["$\sigma_O=$"+str(sigma_surfaces[0]), "$\sigma_0=$"+str(sigma_surfaces[1]), "$\sigma_O>$"+str(sigma_surfaces[2])]

    lonbins = np.arange(0, 361, 5)
    topOfSigmaSurface = np.zeros((12, len(lonbins)-1 ))
    topOfSigmaSurface_corr_CTEMP = np.zeros((12, len(lonbins)-1 ))
    topOfSigmaSurface_corr_sal = np.zeros((12, len(lonbins)-1 ))
    
    for i in range(len(sigma_surfaces)):
        for j in range(12):
            if(surface_type[i] == "equal_to"):
                dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]-tol) &  (dfmg.POT_DENSITY < sigma_surfaces[i]+tol) & (dfmg.SHELF_BREAK_PROFILE) & (dfmg.JULD.dt.month == j+1)
            elif(surface_type[i] == "greater_than"):
                dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]) & (dfmg.SHELF_BREAK_PROFILE) & (dfmg.JULD.dt.month == j+1)
            else:
                raise Exception("surface_type can only be either equal_to or greater_than")

            topOfSigmaSurface[j] = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).apply(find_topOfSigmaSurfaceDepth)
            
            topOfSigmaSurface_corr_CTEMP[j] = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).apply(find_topOfSigmaSurfaceDepth_corr_CTEMP)
            topOfSigmaSurface_corr_sal[j] = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).apply(find_topOfSigmaSurfaceDepth_corr_sal)        

        ax.append(plt.subplot(gs[0, i]))
        print(lonbins[:-1].shape, topOfSigmaSurface.T.shape)
        CF.append(ax[-1].contourf(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface, cmap=cmap[0], vmin=vmin[0], vmax=vmax[0], levels=levs[0], extend='min'))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_title(subplot_titles[i])
        ax[-1].set_xlim(0, 360)
        
        ax.append(plt.subplot(gs[1, i]))
        CF.append(ax[-1].contourf(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface_corr_CTEMP, cmap=cmap[1], vmin=vmin[1], vmax=vmax[1], levels=levs[1], extend='both'))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_xlim(0, 360)
        
        ax.append(plt.subplot(gs[2, i]))
        CF.append(ax[-1].contourf(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface_corr_sal, cmap=cmap[2], vmin=vmin[2], vmax=vmax[2], levels=levs[2], extend='both'))
        ax[-1].set_xlim(0, 360)
        ax[-1].set_xlabel("Longitude")
        

        for j in range(1,4,1):
            if i >0:
                ax[-j].set_yticklabels("")
                ax[-j].set_yticks([])
            else:
                ax[-j].set_ylabel("Months")
                
                
        

    ax.append(plt.subplot(gs[0, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[0], orientation = 'vertical',)
    cbr.ax.set_ylabel("Depth (m)")
    
    ax.append(plt.subplot(gs[1, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[1], orientation = 'vertical',)
    cbr.ax.set_ylabel("CT $^O$C")

    ax.append(plt.subplot(gs[2, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[2], orientation = 'vertical')
    cbr.ax.set_ylabel("Salinity (PSU)")
        
    if save:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        
    dfsel = (dfmg.loc[:, 'LONGITUDE'] > 180)
    dfmg.loc[dfsel, 'LONGITUDE'] = dfmg.loc[dfsel, 'LONGITUDE'] - 360
    
    plt.show()
