from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar
import matplotlib.mlab as mlab

def plot_var_contours_with_distance(df, mask, var, dist=100, bins=5, wd=12, ht=5, varmin=33, varmax=35, levs=[],
                                    colorunit=' ', save=False, savename="Untitled.png", 
                                    zbin=20, xbin=10, zmin=0.0, nmin=3, nlevs=10, ticks=[], cline=[], legend_show=True, fontsize=14):
    if(zmin < 0):
        zlowest = zmin
    else:
        zlowest = df.loc[mask, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
    
    dist_bins = np.arange(0, dist+xbin, xbin)
    
    dist_binned_group = df.loc[mask].groupby(pd.cut(df.loc[mask].DIST_GLINE, dist_bins))
    var_mean = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    potDensity_mean = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    var_count = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    var_sd = np.zeros((len(dist_bins)-1, len(depth_bins)-1))
    i = 0
    for groupList, xGroup in dist_binned_group:
        zGroup = xGroup.groupby(pd.cut(xGroup.DEPTH, depth_bins))
        var_mean[i] = zGroup[var].mean().values
        potDensity_mean[i] = zGroup['gamman'].mean().values
        var_count[i] = zGroup[var].count().values
        var_sd[i] = zGroup[var].std().values
        i += 1
        
    matplotlib.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=(wd, ht))
    zbin_midpoint = depth_bins[:-1] + np.diff(depth_bins)*0.5
    dist_bin_midpoints = dist_bins[:-1] + np.diff(dist_bins)*0.5

    X, Y = np.meshgrid(dist_bin_midpoints, zbin_midpoint)
    if not levs:
        levs = np.linspace(varmin, varmax, nlevs)
    
    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    CF = ax.contourf(X.T[:,:], Y.T[:,:], var_mean[:,:], levs, extend='both')
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Distance from grounding line (km)')
    if(zmin < 0):
        ax.set_ylim(zmin, 0)
    else:
        ax.set_ylim(zlowest, 0)
    if not ticks:
        ticks = list(np.arange(levs[0], levs[-1]+0.2, 0.2))
    if not cline:
        cline = list(np.round(np.arange(27., 29.1, 0.1) , 2))
    CS = ax.contour(X.T[:,:], Y.T[:,:], potDensity_mean[:,:], cline, colors='1', linestyles='solid')
    plt.clabel(CS, cline, colors='1', fontsize=14, fmt='%3.2f')
    cbar1 = fig.colorbar(CF, ax=ax, ticks=ticks, pad=0.01)
    cbar1.set_label(colorunit)
    
    conf_int = var_sd
    conf_int[np.where(var_count < nmin)] = 1e5    
    conf_int[np.where(var_count == 0)] = np.nan    
    conf_int = ma.masked_invalid(conf_int)
    
    print(np.max(conf_int[conf_int < 1e5]))
    CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], 
                         colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
    
    artists, labels = CF2.legend_elements(variable_name="\\sigma")
    labels[-1] = 'count $< $'+str(nmin)
    if(legend_show == True):
        lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.1), fancybox=True, ncol=3)
    ax.set_xlim(0,dist)
    plt.tight_layout()
    if(save== True):
        if(legend_show == True):
            plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(savename, dpi=150)
    plt.show()


def plot_scatter_profilesSet(df, profileSet=[], var='CTEMP', wd=7.48, ht=5, varmin=-2, varmax=5, levs=[], show=True,
                            colorunit='$\\theta^{\circ}$C ', save=False, savename="Untitled.png", fontsize=8,
                            zmin=0.0, ticks=[], cline=[], legend_show=True):
    
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.close(1)
    fig = plt.figure(1, figsize=(wd, ht))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = plt.subplot(gs[0, 0])

    if not profileSet:
        raise ValueError('profileSet cannot be null!')
    selectProfiles = df.PROFILE_NUMBER.isin(profileSet)

    cs = ax.scatter(df.loc[selectProfiles, 'DIST_GLINE'], df.loc[selectProfiles, 'DEPTH'], c=df.loc[selectProfiles, 'CTEMP'])
    cs_gamman = ax.tricontour(df.loc[selectProfiles, 'DIST_GLINE'], df.loc[selectProfiles, 'DEPTH'], df.loc[selectProfiles, 'gamman'], colors='0.5')

    distSortedIndices = np.argsort(df.loc[selectProfiles, 'DIST_GLINE'])
    
    ax.plot(df.loc[selectProfiles, 'DIST_GLINE'].values[distSortedIndices], df.loc[selectProfiles, 'ECHODEPTH'].values[distSortedIndices], linewidth=4, color='k')
    if not cline:
        cline = np.arange(27.8, 28.5, 0.1)
    ax.clabel(cs_gamman, colors='k', fontsize=14, fmt='%3.2f')
    
    colorax = plt.subplot(gs[1,0])
    cbar1 = Colorbar(ax = colorax, mappable = cs, orientation = 'horizontal')
    cbar1.ax.get_children()[0].set_linewidths(5)
    cbar1.set_label(colorunit)

    if save:
        plt.savefig(savename, dpi=300)
    if show:
        plt.show()


def plotProfDist_in_BoundingBox(df, boundingBox=[], var='CTEMP', wd=7.48, ht=5, varmin=-2, varmax=5, levs=[], show=True, plotEcho = False, xlat=False,
                                            colorunit='$\\theta^{\circ}$C ', save=False, savename="Untitled.png", fontsize=8, levels=[],
                                            zmin=0.0, ticks=[], cline=[], legend_show=True, plotTimeHist=False):
    
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.close(1)
    fig = plt.figure(1, figsize=(wd, ht))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = plt.subplot(gs[0, 0])

    if not boundingBox:
        raise ValueError('provide arg boundingBox in fmt [[llcornerx, llcornery], [urcrnrx, urcrnry]] \n With x1,y1 being lower left corner of bounding box, and going counter-clockwise from that point.')
    try:
        leftlonlim = boundingBox[0][0]
        rightlonlim = boundingBox[1][0]
        lowerlatlim = boundingBox[0][1]
        upperlatlim = boundingBox[1][1]
    except:
        raise ValueError('provide arg boundingBox in fmt [[x1,y1], [x2, y2], [x3, y3], [x4, y4]] \n With x1,y1 being lower left corner of bounding box, and going counter-clockwise from that point.')
    
    selectProfiles = (df.LATITUDE >= lowerlatlim) & (df.LATITUDE <= upperlatlim) & (df.LONGITUDE >= leftlonlim) & (df.LONGITUDE <= rightlonlim) & (~df.CTEMP.isnull())

    if xlat:
        dist = df.loc[selectProfiles, 'LATITUDE']
    else:
        dist = df.loc[selectProfiles, 'DIST_GLINE']
    depth = df.loc[selectProfiles, 'DEPTH']
    gamman = df.loc[selectProfiles, 'gamman']

    if xlat:
        ndist = int( (np.max(dist) - np.min(dist) )/ 0.01)
    else:
        ndist = int(df.loc[selectProfiles, 'DIST_GLINE'].max() / 10.)

    dist_grid = np.linspace(np.min(dist), np.max(dist), ndist)
    ndepth = int(np.abs(df.loc[selectProfiles, 'DEPTH'].min()) / 10.)
    depth_grid = np.linspace(df.loc[selectProfiles, 'DEPTH'].min(), 0, ndepth)

    gamman_interpolated = mlab.griddata(dist, depth, gamman, dist_grid, depth_grid, interp='linear')
    cs = ax.scatter(dist, df.loc[selectProfiles, 'DEPTH'], c=df.loc[selectProfiles, 'CTEMP'])
    
    if levels:
        cs_gamman = ax.contour(dist_grid, depth_grid, gamman_interpolated, levels, colors='0.5')
    else:
        cs_gamman = ax.contour(dist_grid, depth_grid, gamman_interpolated, colors='0.5')
        
    distSortedIndices = np.argsort(df.loc[selectProfiles, 'DIST_GLINE'])

    if plotEcho:    
        depthMin = df.loc[selectProfiles].groupby(pd.cut(df.loc[selectProfiles].DIST_GLINE, dist_grid))[["DEPTH"]].min(axis=1).values
        echodepthMin = df.loc[selectProfiles].groupby(pd.cut(df.loc[selectProfiles].DIST_GLINE, dist_grid))[["ECHODEPTH"]].min(axis=1).values
        min_of_depth_echodepth = np.array(list(zip(depthMin, echodepthMin))).min(axis=1)
        ax.plot(dist_grid[:-1], min_of_depth_echodepth, linewidth=4, color='k')
    
    #ax.set_xlim(df.loc[selectProfiles, 'DIST_GLINE'].min()-1, df.loc[selectProfiles, 'DIST_GLINE'].max()+1)

    if not cline:
        cline = np.arange(27.8, 28.5, 0.1)
    ax.clabel(cs_gamman, colors='k', fontsize=14, fmt='%3.2f')
    
    colorax = plt.subplot(gs[1,0])
    cbar1 = Colorbar(ax = colorax, mappable = cs, orientation = 'horizontal')
    cbar1.ax.get_children()[0].set_linewidths(5)
    cbar1.set_label(colorunit)

    if plotTimeHist:
        plt.close(2)
        fig = plt.figure(2, figsize=(wd, ht))
        uniqueProfs = df.loc[selectProfiles].groupby("PROFILE_NUMBER").head(1).index
        df.loc[uniqueProfs, "JULD"].hist()
        plt.show()
    if save:
        plt.savefig(savename, dpi=300)
    if show:
        plt.show()
