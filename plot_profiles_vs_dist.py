from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
from matplotlib.font_manager import FontProperties


def plot_var_contours_with_distance(df, mask, var, dist=100, bins=5, wd=12, ht=5, varmin=33, varmax=35, nlevs=9,
                                    colorunit=' ', save=False, savename="Untitled.png", 
                                    zbin=20, xbin=10, zmin=0.0, nmin=3):
    if(zmin < 0):
        zlowest = zmin
    else:
        zlowest = df.loc[mask, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
    
    dist_bins = np.arange(0, dist+xbin, xbin)
    
    dist_binned_group = df.loc[mask].groupby(pd.cut(df.loc[mask].DIST_GLINE, dist_bins))
    var_mean = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    var_count = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    var_sd = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    i = 0
    for groupList, xGroup in dist_binned_group:
        zGroup = xGroup.groupby(pd.cut(xGroup.DEPTH, depth_bins))
        var_mean[i] = zGroup[var].mean().values
        var_count[i] = zGroup[var].count().values
        var_sd[i] = zGroup[var].std().values
        i += 1
    
    fig, ax = plt.subplots(figsize=(wd, ht))
    zbin_midpoint = depth_bins[:-1] + np.diff(depth_bins)*0.5
    dist_bin_midpoints = dist_bins[:-1] + np.diff(dist_bins)*0.5

    X, Y = np.meshgrid(dist_bin_midpoints, zbin_midpoint)
    levels = np.linspace(varmin, varmax, nlevs)
    
    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    CF = ax.contourf(X.T[:,:], Y.T[:,:], var_mean[:,:], levels)
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Distance from grounding line (km)')
    if(zmin < 0):
        ax.set_ylim(zmin, 0)
    else:
        ax.set_ylim(zlowest, 0)
    cbar1 = fig.colorbar(CF, ax=ax)
    cbar1.set_label(colorunit)
    
    conf_int = var_sd
    conf_int[np.where(var_count < nmin)] = 1e5    
    conf_int[np.where(var_count == 0)] = np.nan    
    conf_int = ma.masked_invalid(conf_int)
    
    levels2 = np.linspace(0,1, 10)
    print(np.max(conf_int[conf_int < 1e5]))
    CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[1e-10, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], 
                         colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
    
    artists, labels = CF2.legend_elements(variable_name="\\sigma")
    labels[-1] = 'count $< $'+str(nmin)    
    plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(1.2, 1))
    ax.set_xlim(0,dist)
    if(save== True):
        plt.savefig(savename)
    plt.show()
