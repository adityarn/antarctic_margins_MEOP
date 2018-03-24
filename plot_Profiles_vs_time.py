from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
from matplotlib.font_manager import FontProperties
import compute_EP_net as computeEPnet
import importlib
#importlib.reload(computeEPnet)
import xarray as xr

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def plot_sal_contours_with_time(df, years=[], bins=5, wd=12, ht=5, varmin=33, varmax=35, nlevs=10,
                                    colorunit='Salinity (PSU)', save=False, savename="Untitled.png", 
                                    zbin=20, zmin=0, nmin=3, depth_max=0.0, levs=[], type=1, integrationDepth=100, plot=True, precip_dir="/media/data/Datasets/AirSeaFluxes/GPCPprecip",
                                    evap_dir="/media/data/Datasets/AirSeaFluxes/WHOIevap"):
    if not years:
        years = np.sort(df.loc[:, 'JULD'].dt.year.unique())
    
    evap_total = []
    for i in range(years[0], years[-1]+1, 1):
        evap_total.append(xr.open_dataset(evap_dir+"/evapr_oaflux_"+str(i)+".nc"))

    precip = xr.open_dataset(precip_dir+"/precip.mon.mean.nc")
    precip_error = xr.open_dataset(precip_dir+"/precip.mon.mean.error.nc")    
    
    if(depth_max < 0):
        zlowest = depth_max
    else:
        zlowest = df.loc[:, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
        
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * len(years)
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, len(years)*12+1, 1)
    

        
    var_binned = np.zeros((len(timeaxis), len(depth_bins)))
    mask = df.loc[:,'JULD'].dt.year.isin(years)
    var_mean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalmean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalstd = np.zeros((len(timeaxis), len(depth_bins)-1))
    rhomean = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_count = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_sd = np.zeros((len(timeaxis), len(depth_bins)-1))
    freshwater_h = np.zeros((len(years), len(depth_bins)-1))
    freshwater_h_error = np.zeros((len(years), len(depth_bins)-1))
    netEP = np.zeros(len(years))
    netEP_error = np.zeros(len(years))
    seaice_fh = np.zeros(len(years))
    seaice_std = np.zeros(len(years))
    
    for i in range(len(years)):
        print(years[i])
        yearmask = df['JULD'].dt.year == years[i]
        count = df.loc[yearmask, 'PSAL_ADJUSTED'].count()
        
        if(count > 0):
            lonmin, lonmax = int(df.loc[yearmask, 'LONGITUDE'].min()), int(df.loc[yearmask, 'LONGITUDE'].max())
            latmin, latmax = int(df.loc[yearmask, 'LATITUDE'].min()), int(df.loc[yearmask, 'LATITUDE'].max())
            evap = evap_total[i]
            netEP[i], netEP_error[i] = computeEPnet.freshwater_flux_compute(evap, precip, precip_error, years[i], lonmin, lonmax, latmin, latmax, plot=False)

            seaice_fh[i], seaice_std[i] = find_sim_freshwater_h(years[i], lonmin, lonmax, latmin, latmax)

            for j in range(12):
                monthmask = df['JULD'].dt.month == j+1
                timeSlice_df = df.loc[yearmask & monthmask]
                var_mean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.mean().values
                abssalmean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.mean().values
                abssalstd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.std().values
                rhomean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).DENSITY_INSITU.mean().values
                var_count[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.count().values
                var_sd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.std().values

            if(type == 1):
                for b in range(len(depth_bins)-1):
                    s,e = i*12, i*12+12
                    if(np.isnan(var_mean[s:e, b]).all()):
                        continue
                    else:
                        min_ind = np.nanargmin(var_mean[s : e, b])
                        max_ind = np.nanargmax(var_mean[s : e, b])
                                                
                        h_w = abs(depth_bins[1] - depth_bins[0])
                        freshwater_h[i][b] = h_w * ( (rhomean[s:e, b][max_ind] * abssalmean[s:e, b][max_ind]) / (rhomean[s:e, b][min_ind] * abssalmean[s:e, b][min_ind])  - 1 ) * 1e3
                        freshwater_h_error[i][b] = h_w * abs(freshwater_h[i][b]) * np.sqrt( (abssalstd[s:e,b][max_ind] / abssalmean[s:e,b][max_ind])**2 +  (abssalstd[s:e,b][min_ind] / abssalmean[s:e,b][min_ind])**2)
                        ## if(b == len(depth_bins)-2):
                        ##     freshwater_h[i][b] += -netEP[i]
                        ##     freshwater_h[i][b] += -seaice_fh
                        ##     print(netEP[i])

            if(type == 2):
                s,e = i*12, i*12+12
                b_max = np.argmin(np.abs(-depth_bins - integrationDepth))
                depthIntsal = np.zeros(12)
                depthIntRho = np.zeros(12)
                for k in range(12):
                    depthIntsal[k] = abssalmean[s:e, ::-1][k, 0:b_max+1].mean()
                    depthIntRho[k] = rhomean[s:e, ::-1][k, 0:b_max+1].mean()
                if(np.isnan(depthIntsal).all()):
                    continue
                else:
                    min_ind = np.nanargmin(depthIntsal)
                    max_ind = np.nanargmax(depthIntsal)
                    h_w = abs(depth_bins[::-1][b_max])

                    freshwater_h[i][0] = h_w * ( (depthIntRho[max_ind] * depthIntsal[max_ind]) / (depthIntRho[min_ind] * depthIntsal[min_ind])  - 1 ) * 1e3
                    freshwater_h[i][1:] = np.nan
        else:
            var_mean[i*12 : i*12+13] = np.nan



    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    #fig.subplots_adjust(hspace=1.3)
    if(plot == True):
        fig, ax = plt.subplots(nrows=1, ncols=1,squeeze=True, figsize=(wd, ht))
        year_ax = ax.twiny()

        year_ax.set_frame_on(True)
        year_ax.patch.set_visible(False)
        year_ax.xaxis.set_ticks_position('bottom')
        year_ax.xaxis.set_label_position('bottom')
        year_ax.spines['bottom'].set_position(('outward', 30))

        zbin_midpoint = depth_bins[:-1] + np.diff(depth_bins)*0.5
        #zbin_midpoint = np.insert(zbin_midpoint, len(zbin_midpoint), 0)
        timeaxis_midpoint = timeaxis[:-1] + np.diff(timeaxis)*0.5
        X, Y = np.meshgrid(timeaxis, zbin_midpoint) #depth_bins[1:])
        if not levs:
            levs = np.linspace(varmin, varmax, nlevs)
        CF = ax.contourf(X.T[:, :], Y.T[:, :], var_mean[:, :], levs, zorder=0, extend='both')
        ax.set_xticks(timeaxis)
        ax.set_xticklabels(timeaxis_ticklabel)
        ax.set_ylabel('Depth (m)')

        if(depth_max < 0):
            ax.set_ylim(depth_max, 0)
        year_ax.set_xticks(np.arange(1,len(timeaxis)+1, 12))
        year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
        year_ax.set_xlim(1, timeaxis[-1])

        #cbaxes = fig.add_axes([1.005, 0.075, 0.02, 0.885]) 
        #cbar1 = fig.colorbar(CF, cax=cbaxes)
        cbar1 = fig.colorbar(CF, ax=[ax, year_ax], pad=0.015)
        #cbar1.set_label(colorunit, labelpad=4, y=0.5)
        cbar1.set_label(colorunit)

        #conf_int = 1.96*var_sd/np.sqrt(var_count)
        conf_int = var_sd
        conf_int[np.where(var_count < nmin)] = 1e5    
        conf_int[np.where(var_count == 0)] = np.nan    
        conf_int = ma.masked_invalid(conf_int)

        print(np.mean(conf_int[conf_int < 1e5]), np.max(conf_int[conf_int < 1e5]))

        CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
        #fontP = FontProperties()
        #fontP.set_size('small')
        #legend([plot1], "title", prop=fontP)
        artists, labels = CF2.legend_elements(variable_name="\\sigma")
        labels[-1] = 'count $< $'+str(nmin)
        lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.19), fancybox=True, ncol=3)

        if(zmin != 0):
            ax.set_ylim(zmin, 0)
        else:
            ax.set_ylim(zlowest, 0)

        if(save== True):
            plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
        
    return [ [freshwater_h, freshwater_h_error] , \
             [netEP, netEP_error] , \
             [seaice_fh, seaice_std] ] , depth_bins


def plot_CT_contours_with_time(df, years=[], bins=5, wd=12, ht=5, varmin=-3, varmax=1, nlevs=10,
                                    colorunit='Pot. temp. $\\theta^o$C', save=False, savename="Untitled.png", 
                                    zbin=20, zmin=0, nmin=3, depth_max=0.0, levs=[]):
    if(depth_max < 0):
        zlowest = depth_max
    else:
        zlowest = df.loc[:, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
    if not years:
        years = np.sort(df.loc[:, 'JULD'].dt.year.unique())
        
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * len(years)
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, len(years)*12+1, 1)
    
    fig, ax = plt.subplots(figsize=(wd, ht))
    year_ax = ax.twiny()

        
    var_binned = np.zeros((len(timeaxis), len(depth_bins)))
    mask = df.loc[:,'JULD'].dt.year.isin(years)
    var_mean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalmean = np.zeros((len(timeaxis), len(depth_bins)-1))
    rhomean = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_count = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_sd = np.zeros((len(timeaxis), len(depth_bins)-1))
    freshwater_h = np.zeros((len(years), len(depth_bins)-1))
    for i in range(len(years)):
        yearmask = df['JULD'].dt.year == years[i]
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            timeSlice_df = df.loc[yearmask & monthmask]
            var_mean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.mean().values
            var_count[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.count().values
            var_sd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.std().values



    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    #fig.subplots_adjust(hspace=1.3)
    year_ax.set_frame_on(True)
    year_ax.patch.set_visible(False)
    year_ax.xaxis.set_ticks_position('bottom')
    year_ax.xaxis.set_label_position('bottom')
    year_ax.spines['bottom'].set_position(('outward', 30))

    zbin_midpoint = depth_bins[:-1] + np.diff(depth_bins)*0.5
    #zbin_midpoint = np.insert(zbin_midpoint, len(zbin_midpoint), 0)
    timeaxis_midpoint = timeaxis[:-1] + np.diff(timeaxis)*0.5
    X, Y = np.meshgrid(timeaxis, zbin_midpoint) #depth_bins[1:])
    if not levs:
        levs = np.linspace(varmin, varmax, nlevs)
    CF = ax.contourf(X.T[:, :], Y.T[:, :], var_mean[:, :], levs, zorder=0, extend='both')
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    ax.set_ylabel('Depth (m)')
    
    if(depth_max < 0):
        ax.set_ylim(depth_max, 0)
    year_ax.set_xticks(np.arange(1,len(timeaxis)+1, 12))
    year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
    year_ax.set_xlim(1, timeaxis[-1])

    #cbaxes = fig.add_axes([1.005, 0.075, 0.02, 0.885]) 
    #cbar1 = fig.colorbar(CF, cax=cbaxes)
    cbar1 = fig.colorbar(CF, ax=[ax, year_ax], pad=0.015)
    #cbar1.set_label(colorunit, labelpad=4, y=0.5)
    cbar1.set_label(colorunit)

    #conf_int = 1.96*var_sd/np.sqrt(var_count)
    conf_int = var_sd
    conf_int[np.where(var_count < nmin)] = 1e5    
    conf_int[np.where(var_count == 0)] = np.nan    
    conf_int = ma.masked_invalid(conf_int)

    print(np.mean(conf_int[conf_int < 1e5]), np.max(conf_int[conf_int < 1e5]))

    CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
    #fontP = FontProperties()
    #fontP.set_size('small')
    #legend([plot1], "title", prop=fontP)
    artists, labels = CF2.legend_elements(variable_name="\\sigma")
    labels[-1] = 'count $< $'+str(nmin)
    plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(1.12, 1))
        
    if(zmin != 0):
        ax.set_ylim(zmin, 0)
    else:
        ax.set_ylim(zlowest, 0)
    
    #plt.tight_layout();    
    if(save== True):
        plt.savefig(savename, dpi=150)
    plt.show()


def find_sim_freshwater_h(year, lonmin, lonmax, latmin, latmax, plot=False):
    
    def find_lonlat_indices_seaice(seaice, time, lon, lat):
        time_ind = np.asscalar(np.argmin(np.abs(seaice.time - time )))
        return time_ind, np.asscalar(np.argmin(np.abs(seaice.lon - lon))), np.asscalar(np.argmin(np.abs(seaice.lat - lat)))
    
    seaice = xr.open_dataset("/media/data/Datasets/SeaIce/ocn_ana_2D_ll.nc")
    timeind, lonminind, latminind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmin, latmin)
    _, lonmaxind, latmaxind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmax, latmax)
    sim_monthly_area_average = np.zeros(12)
    sim_monthly_area_std = np.zeros(12)
    for i in range(12):
        sim_monthly_area_average[i] = seaice.sim.isel(time=timeind+i, lon=slice(lonminind, lonmaxind), 
                                                      lat=slice(latminind, latmaxind)).mean()
        sim_monthly_area_std[i] = seaice.sim.isel(time=timeind+i, lon=slice(lonminind, lonmaxind), 
                                                      lat=slice(latminind, latmaxind)).std()
    
    if(plot==True):
        plt.plot(np.arange(1,13,1), sim_monthly_area_average)
        plt.show()
    if(np.isnan(sim_monthly_area_average).all() == True):
        minind, maxind = 0, 0
    else:
        minind, maxind = np.nanargmin(sim_monthly_area_average), np.nanargmax(sim_monthly_area_average)

    if(np.isnan(sim_monthly_area_std).all() == True):
        fh_std = np.nan
    else:
        fh_std = np.nanmax(sim_monthly_area_std)
        
    mass_diff = sim_monthly_area_average[maxind] - sim_monthly_area_average[minind]
    
    fh = mass_diff / 1030. * 1e3
    
    
    return fh, fh_std
