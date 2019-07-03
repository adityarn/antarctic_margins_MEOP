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


def plot_slope_sigma0_surfaces_CF_climatology(dfmg, sigma_surfaces=[27.67, 27.74, 27.86], surface_type=["equal_to", "equal_to", "greater_than"], save=False, savename="Untitled.png", tol=0.01, figno=1, wd=190/25.4, ht=230/25.4):
    plt.close(figno)
    plt.figure(figno, figsize=(wd, ht))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1,1,1,0.01, 0.1], wspace=0.01, hspace=0.01)
    
    vmin = [-1500, -1.5, 34.4]
    vmax= [0, 1.5, 34.8]
    levs = [np.arange(vmin[0], vmax[0]+0.1, 10), np.arange(vmin[1], vmax[1]+0.01, 0.05) , 
            np.arange(vmin[2], vmax[2]+0.01, 0.005)]
    extend = ['min', 'both']

    cmap = ['viridis', 'coolwarm', 'viridis']
    CF = []
    ax = []
    subplot_titles = ["$\sigma_O=$"+str(sigma_surfaces[0]), "$\sigma_0=$"+str(sigma_surfaces[1]), "$\sigma_O=$"+str(sigma_surfaces[2])]

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
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(0,13,1), topOfSigmaSurface, cmap=cmap[0], vmin=vmin[0], vmax=vmax[0], ))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_title(subplot_titles[i])
        ax[-1].set_xlim(0, 360)
        ax[-1].set_yticks(np.arange(1,12, 2))
        
        ax.append(plt.subplot(gs[1, i]))
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(0,13,1), topOfSigmaSurface_corr_CTEMP, cmap=cmap[1], vmin=vmin[1], vmax=vmax[1]))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_xlim(0, 360)
        ax[-1].set_yticks(np.arange(1,12, 2))
        
        ax.append(plt.subplot(gs[2, i]))
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(0,13,1), topOfSigmaSurface_corr_sal, cmap=cmap[2], vmin=vmin[2], vmax=vmax[2], ))
        ax[-1].set_xlim(0, 360)
        ax[-1].set_xlabel("Longitude")
        ax[-1].set_yticks(np.arange(1,12, 2))
        

        for j in range(1,4,1):
            if i >0:
                ax[-j].set_yticklabels("")
                #ax[-j].set_yticks([])
            else:
                ax[-j].set_ylabel("Months")
                
                
        

    ax.append(plt.subplot(gs[0, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[0], orientation = 'vertical', extend='min')
    cbr.ax.set_ylabel("Depth (m)")
    
    ax.append(plt.subplot(gs[1, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[1], orientation = 'vertical', extend='both')
    cbr.ax.set_ylabel("CT $^O$C")

    ax.append(plt.subplot(gs[2, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[2], orientation = 'vertical', extend = 'both')
    cbr.ax.set_ylabel("Salinity (PSU)")
        
    if save:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        
    
    plt.show()






def plot_slope_sigma0_surfaces_variability_mean(dfmg, sigma_surfaces=[27.67, 27.74, 27.86], surface_type=["equal_to", "equal_to", "greater_than"], save=False, savename="Untitled.png", tol=0.01, figno=1, wd=190/25.4, ht=230/25.4):
    plt.close(figno)
    plt.figure(figno, figsize=(wd, ht))
    gs = gridspec.GridSpec(4, 5, width_ratios=[1,1,1,0.01, 0.1], height_ratios = [1,1,1,0.5], wspace=0.01, hspace=0.01)
    
    vmin = [-1500, -1.5, 34.4]
    vmax= [0, 1.5, 34.8]
    levs = [np.arange(vmin[0], vmax[0]+0.1, 10), np.arange(vmin[1], vmax[1]+0.01, 0.05) , 
            np.arange(vmin[2], vmax[2]+0.01, 0.005)]
    extend = ['min', 'both']

    cmap = ['viridis', 'coolwarm', 'viridis']
    CF = []
    ax = []
    subplot_titles = ["$\sigma_O=$"+str(sigma_surfaces[0]), "$\sigma_0=$"+str(sigma_surfaces[1]), "$\sigma_O=$"+str(sigma_surfaces[2])]

    lonbins = np.arange(0, 361, 5)
    topOfSigmaSurface_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_std = np.zeros(len(lonbins)-1 )
    
    topOfSigmaSurface_corr_CTEMP_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_corr_CTEMP_std = np.zeros(len(lonbins)-1 )
    
    topOfSigmaSurface_corr_sal_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_corr_sal_std = np.zeros(len(lonbins)-1 )    
    ylabel = ["Depth (m)", "CT$^\circ$C", "Salinity (PSU)", "Count"]
    subplot_labels = [["(a)", "(e)", "(i)"], ["(b)", "(f)", "(j)"], ["(c)", "(g)", "(k)"], ["(d)", "(h)", "(l)"]]
    for i in range(len(sigma_surfaces)):
        dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]-tol) &  (dfmg.POT_DENSITY < sigma_surfaces[i]+tol) & (dfmg.SHELF_BREAK_PROFILE)
        # if(surface_type[i] == "equal_to"):
        #     dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]-tol) &  (dfmg.POT_DENSITY < sigma_surfaces[i]+tol) & (dfmg.SHELF_BREAK_PROFILE)
        # elif(surface_type[i] == "greater_than"):
        #     dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]) & (dfmg.SHELF_BREAK_PROFILE)
        # else:
        #     raise Exception("surface_type can only be either equal_to or greater_than")

        topOfSigmaSurface_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).DEPTH.mean()        
        topOfSigmaSurface_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).DEPTH.std()

        topOfSigmaSurface_corr_CTEMP_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).CTEMP.mean()
        topOfSigmaSurface_corr_CTEMP_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).CTEMP.std()

        topOfSigmaSurface_corr_sal_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).PSAL_ADJUSTED.mean()        
        topOfSigmaSurface_corr_sal_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).PSAL_ADJUSTED.std()

        ax.append(plt.subplot(gs[0, i]))

        ax[-1].plot(lonbins[:-1], topOfSigmaSurface_mean.values, marker=".", color="k", linewidth=0)
        yerr=topOfSigmaSurface_std.values
        ax[-1].fill_between(lonbins[:-1], topOfSigmaSurface_mean.values-yerr, topOfSigmaSurface_mean.values+yerr, facecolor="0.25", edgecolor="0.5", alpha=0.5 )
        
        ax[-1].set_xlim(0, 360)
        if(i < 2):
            ax[-1].set_ylim(-700, 0)
            ax[-1].grid(linestyle=":")
            ax[-1].text(10, -650, subplot_labels[0][i])
        else:
            ax[-1].set_ylim(-2000, 0)
            axr = ax[-1].twinx()
            axr.plot(lonbins[:-1], topOfSigmaSurface_mean.values, marker=".", color="k", linewidth=0)
            axr.set_ylim(-2000, 0)
            axr.text(25, -650/700.*2e3, subplot_labels[0][i])
            axr.set_ylabel("Depth (m)")
            ax[-1].grid(linestyle=":")
        #ax[-1].set_xticklabels("")
        #ax[-1].set_xticks([])
        ax[-1].set_title(subplot_titles[i])
            
        
        ax.append(plt.subplot(gs[1, i]))
        ax[-1].plot(lonbins[:-1], topOfSigmaSurface_corr_CTEMP_mean.values, marker=".", color="r", linewidth=0)
        yerr= topOfSigmaSurface_corr_CTEMP_std.values
        ax[-1].fill_between(lonbins[:-1], topOfSigmaSurface_corr_CTEMP_mean.values - yerr, topOfSigmaSurface_corr_CTEMP_mean.values + yerr, facecolor="coral", edgecolor="r", alpha=0.5)
        
        #ax[-1].set_xticklabels("")
        #ax[-1].set_xticks([])
        ax[-1].set_xlim(0, 360)
        ax[-1].set_ylim(-2, 2)
        ax[-1].grid(linestyle=":")
        ax[-1].text(10, 1.5, subplot_labels[1][i])
        
        ax.append(plt.subplot(gs[2, i]))
        ax[-1].plot(lonbins[:-1], topOfSigmaSurface_corr_sal_mean.values, marker=".", color="b", linewidth=0)
        yerr=topOfSigmaSurface_corr_sal_std.values
        ax[-1].fill_between(lonbins[:-1], topOfSigmaSurface_corr_sal_mean.values - yerr, topOfSigmaSurface_corr_sal_mean.values + yerr, facecolor="skyblue", edgecolor="b", alpha=0.5) 
        ax[-1].set_xlim(0, 360)
        ax[-1].set_xlabel("Longitude")
        ax[-1].set_ylim(34.3, 34.85)
        ax[-1].grid(linestyle=":")
        ax[-1].text(10, 34.8, subplot_labels[2][i])
        #ax[-1].set_xticklabels("")

        ax.append(plt.subplot(gs[3, i]) )
        count = dfmg[dfsel].CTEMP.groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins) ).count()
        ax[-1].fill_between(lonbins[:-1], 0, count, color="k", alpha=0.5)
        ax[-1].plot(lonbins[:-1], count, linewidth=0, marker="o", color="k", markersize=1)
        ax[-1].set_ylim(1e-1, 4e3)
        ax[-1].set_yscale("log")
        ax[-1].text(10, 1e3, subplot_labels[3][i])
        ax[-1].set_yticks([ 1e0, 1e1, 1e2, 1e3])
        
        for j in range(1,5,1):
            if i>0:
                pass
                #ax[-j].set_yticklabels("")
                #ax[-j].set_yticks([])
            else:
                ax[-j].set_ylabel(ylabel[::-1][j-1])
                
                
                
    if save:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        
    
    plt.show()










def plot_slope_sigma0_surfaces_CF(dfmg, sigma_surfaces=[27.67, 27.74, 27.86], surface_type=["equal_to", "equal_to", "greater_than"], save=False, savename="Untitled.png", tol=0.01, figno=1, wd=190/25.4, ht=230/25.4):
    dfsel = (dfmg.loc[:, 'LONGITUDE'] < 0)
    dfmg.loc[dfsel, 'LONGITUDE'] = dfmg.loc[dfsel, 'LONGITUDE'] + 360
    plt.close(figno)
    plt.figure(figno, figsize=(wd, ht))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1,1,1,0.01, 0.1], wspace=0.01, hspace=0.01)
    
    vmin = [-1500, -1.5, 34.4]
    vmax= [0, 1.5, 34.8]
    levs = [np.arange(vmin[0], vmax[0]+0.1, 10), np.arange(vmin[1], vmax[1]+0.01, 0.05) , 
            np.arange(vmin[2], vmax[2]+0.01, 0.005)]
    extend = ['min', 'both']

    cmap = ['viridis', 'RdBu_r', 'viridis']
    CF = []
    ax = []
    subplot_titles = ["$\sigma_O=$"+str(sigma_surfaces[0]), "$\sigma_0=$"+str(sigma_surfaces[1]), "$\sigma_O=$"+str(sigma_surfaces[2])]

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
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface, cmap=cmap[0], vmin=vmin[0], vmax=vmax[0], ))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_title(subplot_titles[i])
        ax[-1].set_xlim(0, 360)
        
        ax.append(plt.subplot(gs[1, i]))
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface_corr_CTEMP, cmap=cmap[1], vmin=vmin[1], vmax=vmax[1]))
        ax[-1].set_xticklabels("")
        ax[-1].set_xticks([])
        ax[-1].set_xlim(0, 360)
        
        ax.append(plt.subplot(gs[2, i]))
        CF.append(ax[-1].pcolormesh(lonbins[:-1], np.arange(1,13,1), topOfSigmaSurface_corr_sal, cmap=cmap[2], vmin=vmin[2], vmax=vmax[2], ))
        ax[-1].set_xlim(0, 360)
        ax[-1].set_xlabel("Longitude")
        

        for j in range(1,4,1):
            if i >0:
                ax[-j].set_yticklabels("")
                ax[-j].set_yticks([])
            else:
                ax[-j].set_ylabel("Months")
                
                
        

    ax.append(plt.subplot(gs[0, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[0], orientation = 'vertical', extend='min')
    cbr.ax.set_ylabel("Depth (m)")
    
    ax.append(plt.subplot(gs[1, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[1], orientation = 'vertical', extend='both')
    cbr.ax.set_ylabel("CT $^O$C")

    ax.append(plt.subplot(gs[2, 4]))
    cbr = Colorbar(ax = ax[-1], mappable = CF[2], orientation = 'vertical', extend = 'both')
    cbr.ax.set_ylabel("Salinity (PSU)")
        
    if save:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        
    dfsel = (dfmg.loc[:, 'LONGITUDE'] > 180)
    dfmg.loc[dfsel, 'LONGITUDE'] = dfmg.loc[dfsel, 'LONGITUDE'] - 360
    
    plt.show()






def plot_slope_sigma0_surfaces_variability(dfmg, sigma_surfaces=[27.67, 27.74, 27.86], surface_type=["equal_to", "equal_to", "greater_than"], save=False, savename="Untitled.png", tol=0.01, figno=1, wd=190/25.4, ht=230/25.4):
    plt.close(figno)
    plt.figure(figno, figsize=(wd, ht))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1,1,1,0.01, 0.1], wspace=0.01, hspace=0.01)
    
    vmin = [-1500, -1.5, 34.4]
    vmax= [0, 1.5, 34.8]
    levs = [np.arange(vmin[0], vmax[0]+0.1, 10), np.arange(vmin[1], vmax[1]+0.01, 0.05) , 
            np.arange(vmin[2], vmax[2]+0.01, 0.005)]
    extend = ['min', 'both']

    cmap = ['viridis', 'coolwarm', 'viridis']
    CF = []
    ax = []
    subplot_titles = ["$\sigma_O=$"+str(sigma_surfaces[0]), "$\sigma_0=$"+str(sigma_surfaces[1]), "$\sigma_O=$"+str(sigma_surfaces[2])]

    lonbins = np.arange(0, 361, 5)
    topOfSigmaSurface_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_std = np.zeros(len(lonbins)-1 )
    
    topOfSigmaSurface_corr_CTEMP_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_corr_CTEMP_std = np.zeros(len(lonbins)-1 )
    
    topOfSigmaSurface_corr_sal_mean = np.zeros(len(lonbins)-1 )
    topOfSigmaSurface_corr_sal_std = np.zeros(len(lonbins)-1 )    
    ylabel = ["Depth (m)", "CT$^\circ$C", "Salinity (PSU)"]
    subplot_labels = [["(a)", "(d)", "(g)"], ["(b)", "(e)", "(h)"], ["(c)", "(f)", "(i)"]]
    for i in range(len(sigma_surfaces)):
        dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]-tol) &  (dfmg.POT_DENSITY < sigma_surfaces[i]+tol) & (dfmg.SHELF_BREAK_PROFILE)
        # if(surface_type[i] == "equal_to"):
        #     dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]-tol) &  (dfmg.POT_DENSITY < sigma_surfaces[i]+tol) & (dfmg.SHELF_BREAK_PROFILE)
        # elif(surface_type[i] == "greater_than"):
        #     dfsel = (dfmg.POT_DENSITY > sigma_surfaces[i]) & (dfmg.SHELF_BREAK_PROFILE)
        # else:
        #     raise Exception("surface_type can only be either equal_to or greater_than")

        topOfSigmaSurface_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).DEPTH.mean()        
        topOfSigmaSurface_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).DEPTH.std()

        topOfSigmaSurface_corr_CTEMP_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).CTEMP.mean()
        topOfSigmaSurface_corr_CTEMP_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).CTEMP.std()

        topOfSigmaSurface_corr_sal_mean = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).PSAL_ADJUSTED.mean()        
        topOfSigmaSurface_corr_sal_std = dfmg.loc[dfsel].groupby(pd.cut(dfmg[dfsel].LONGITUDE, lonbins )).PSAL_ADJUSTED.std()

        ax.append(plt.subplot(gs[0, i]))

        yerr=topOfSigmaSurface_std.values
        ax[-1].plot(lonbins[:-1], yerr, marker=".", color="k", linewidth=0.2)
        
        ax[-1].set_xlim(0, 360)
        ax[-1].grid(linestyle=":")
        ax[-1].set_xticklabels("")
        #ax[-1].set_xticks([])
        ax[-1].set_title(subplot_titles[i])
            
        
        ax.append(plt.subplot(gs[1, i]))
        yerr= topOfSigmaSurface_corr_CTEMP_std.values        
        ax[-1].plot(lonbins[:-1], yerr, marker=".", color="r", linewidth=0.2)

        ax[-1].set_xticklabels("")
        #ax[-1].set_xticks([])
        ax[-1].set_xlim(0, 360)
        #ax[-1].set_ylim(-2, 2)
        ax[-1].grid(linestyle=":")
        #ax[-1].text(10, 1.5, subplot_labels[1][i])
        
        ax.append(plt.subplot(gs[2, i]))
        yerr=topOfSigmaSurface_corr_sal_std.values        
        ax[-1].plot(lonbins[:-1], yerr, marker=".", color="b", linewidth=0.2)
        ax[-1].set_xlim(0, 360)
        ax[-1].set_xlabel("Longitude")
        #ax[-1].set_ylim(34.3, 34.85)
        ax[-1].grid(linestyle=":")
        #ax[-1].text(10, 34.8, subplot_labels[2][i])
        
        for j in range(1,4,1):
            if i>0:
                pass
                #ax[-j].set_yticklabels("")
                #ax[-j].set_yticks([])
            else:
                ax[-j].set_ylabel(ylabel[::-1][j-1])
                
                
                
    if save:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
        
    
    plt.show()
