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
importlib.reload(computeEPnet)
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

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
                                    evap_dir="/media/data/Datasets/AirSeaFluxes/WHOIevap", clim=False, lonmin=np.nan, lonmax=np.nan, fontsize=14, show_legend=False):
    matplotlib.rcParams.update({'font.size': fontsize})

    if not years:
        years = np.sort(df.loc[:, 'JULD'].dt.year.unique())
        
    if(clim == False):
        iter_range = len(years)
        evap_year_start, evap_year_end = years[0], years[-1]
    else:
        iter_range = 1
        evap_year_start, evap_year_end = 2004, 2015
    
    evap_total = []
    for i in range(evap_year_start, evap_year_end+1, 1):
        evap_total.append(xr.open_dataset(evap_dir+"/evapr_oaflux_"+str(i)+".nc"))

    precip = xr.open_dataset(precip_dir+"/precip.mon.mean.nc")
    precip_error = xr.open_dataset(precip_dir+"/precip.mon.mean.error.nc")    
    
    if(depth_max < 0):
        zlowest = depth_max
    else:
        zlowest = df.loc[:, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
        
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * iter_range
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, iter_range*12+1, 1)
    
    var_binned = np.zeros((len(timeaxis), len(depth_bins)))
    mask = df.loc[:,'JULD'].dt.year.isin(years)
    var_mean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalmean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalstd = np.zeros((len(timeaxis), len(depth_bins)-1))
    rhomean = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_count = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_sd = np.zeros((len(timeaxis), len(depth_bins)-1))
    freshwater_h = np.zeros((iter_range, len(depth_bins)-1))
    freshwater_h_error = np.zeros((iter_range, len(depth_bins)-1))
    netEP = np.zeros(iter_range)
    netEP_error = np.zeros(iter_range)
    seaice_fh = np.zeros(iter_range)
    seaice_std = np.zeros(iter_range)
    
    for i in range(iter_range):

        if(clim == False):
            yearmask = df['JULD'].dt.year == years[i]
        else:
            yearmask = df['JULD'].dt.year.isin(years)
        count = df.loc[yearmask, 'PSAL_ADJUSTED'].count()
        
        if(count > 0):
            if(np.isnan(lonmin) & np.isnan(lonmax)):
                lonmin, lonmax = int(df.loc[yearmask, 'LONGITUDE'].min()), int(df.loc[yearmask, 'LONGITUDE'].max())
            latmin, latmax = int(df.loc[yearmask, 'LATITUDE'].min()), int(df.loc[yearmask, 'LATITUDE'].max())
            print(lonmin, lonmax, latmin, latmax)
            evap = evap_total[i]
            if(clim == False):
                netEP[i], netEP_error[i] = computeEPnet.freshwater_flux_compute(PEclim, lonmin, lonmax, latmin, latmax, plot=False, year=years[i])
                seaice_fh[i], seaice_std[i] = find_sim_freshwater_h(lonmin, lonmax, latmin, latmax, year=years[i])
            else:
                netEP[i], netEP_error[i] = computeEPnet.freshwater_flux_compute(evap_total, precip, precip_error, lonmin, lonmax, latmin, latmax, clim=True, plot=False, year=[evap_year_start, evap_year_end])
                seaice_fh[i], seaice_std[i] = find_sim_freshwater_h(lonmin, lonmax, latmin, latmax, clim=True)


            for j in range(12):
                monthmask = df['JULD'].dt.month == j+1
                timeSlice_df = df.loc[yearmask & monthmask]
                var_mean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.mean().values
                abssalmean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.mean().values
                abssalstd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.std().values
                rhomean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).DENSITY_INSITU.mean().values
                var_count[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.count().values
                var_sd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.std().values

            if(type == 0): ## type 0 does not compute with rho, it is h_{bin} ( \frac{SA_max}{SA_min} - 1 )
                for b in range(len(depth_bins)-1):
                    s,e = i*12, i*12+12
                    if(np.isnan(var_mean[s:e, b]).all()):
                        continue
                    else:
                        min_ind = np.nanargmin(abssalmean[s : e, b])
                        max_ind = np.nanargmax(abssalmean[s : e, b])
                                                
                        h_w = abs(depth_bins[1] - depth_bins[0])

                        freshwater_h[i][b] = h_w * (abssalmean[s:e, b][max_ind] - abssalmean[s:e, b][min_ind]) * 1e3
                        freshwater_h_error[i][b] = abs(freshwater_h[i][b]) * np.sqrt( (abssalstd[s:e,b][max_ind] / abssalmean[s:e,b][max_ind])**2 +  (abssalstd[s:e,b][min_ind] / abssalmean[s:e,b][min_ind])**2)
                
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
                        freshwater_h_error[i][b] = abs(freshwater_h[i][b]) * np.sqrt( (abssalstd[s:e,b][max_ind] / abssalmean[s:e,b][max_ind])**2 +  (abssalstd[s:e,b][min_ind] / abssalmean[s:e,b][min_ind])**2)

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

        if(clim == False):
            year_ax = ax.twiny()
            year_ax.set_frame_on(True)
            year_ax.patch.set_visible(False)
            year_ax.xaxis.set_ticks_position('bottom')
            year_ax.xaxis.set_label_position('bottom')
            year_ax.spines['bottom'].set_position(('outward', 30))
            year_ax.set_xticks(np.arange(1,len(timeaxis)+1, 12))
            year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
            year_ax.set_xlim(1, timeaxis[-1])
            

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

        #cbaxes = fig.add_axes([1.005, 0.075, 0.02, 0.885]) 
        #cbar1 = fig.colorbar(CF, cax=cbaxes)
        if(clim == False):
            cbar1 = fig.colorbar(CF, ax=[ax, year_ax], pad=0.015)
        else:
            cbar1 = fig.colorbar(CF, ax=ax, pad=0.015)
        #cbar1.set_label(colorunit, labelpad=4, y=0.5)
        cbar1.set_label(colorunit)

        #conf_int = 1.96*var_sd/np.sqrt(var_count)
        conf_int = var_sd
        conf_int[np.where(var_count < nmin)] = 1e5    
        conf_int[np.where(var_count == 0)] = np.nan    
        conf_int = ma.masked_invalid(conf_int)

        CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
        #fontP = FontProperties()
        #fontP.set_size('small')
        #legend([plot1], "title", prop=fontP)
        artists, labels = CF2.legend_elements(variable_name="\\sigma")
        labels[-1] = 'count $< $'+str(nmin)
        
        if(show_legend == True):
            if(clim == False):
                lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.19), fancybox=True, ncol=3)
            if(clim == True):
                lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.1), fancybox=True, ncol=3)
            
        if(zmin != 0):
            ax.set_ylim(zmin, 0)
        else:
            ax.set_ylim(zlowest, 0)

        if(save== True):
            if(show_legend == True):
                plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                plt.savefig(savename, dpi=150)
        plt.show()
        
    return [ [freshwater_h, freshwater_h_error] , \
             [netEP, netEP_error] , \
             [seaice_fh, seaice_std] ] , depth_bins


def plot_CT_contours_with_time(df, years=[], bins=5, wd=12, ht=5, varmin=-3, varmax=1, nlevs=10,
                                    colorunit='Pot. temp. $\\theta^o$C', save=False, savename="Untitled.png", 
                                    zbin=20, zmin=0, nmin=3, depth_max=0.0, levs=[], clim=True, show_legend=False, fontsize=14):
    matplotlib.rcParams.update({'font.size': fontsize})        
    if(depth_max < 0):
        zlowest = depth_max
    else:
        zlowest = df.loc[:, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
    if not years:
        years = np.sort(df.loc[:, 'JULD'].dt.year.unique())
    if(clim == True):
        iter_range = 1
    else:
        iter_range = len(years)
        
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * iter_range
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, iter_range*12+1, 1)
    
    fig, ax = plt.subplots(figsize=(wd, ht))


        
    var_binned = np.zeros((len(timeaxis), len(depth_bins)))
    mask = df.loc[:,'JULD'].dt.year.isin(years)
    var_mean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalmean = np.zeros((len(timeaxis), len(depth_bins)-1))
    rhomean = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_count = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_sd = np.zeros((len(timeaxis), len(depth_bins)-1))

    for i in range(iter_range):
        if(clim == False):
            yearmask = df['JULD'].dt.year == years[i]
        else:
            yearmask = [True] * len(df)
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            timeSlice_df = df.loc[yearmask & monthmask]
            var_mean[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.mean().values
            var_count[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.count().values
            var_sd[i*12+j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).CTEMP.std().values



    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    #fig.subplots_adjust(hspace=1.3)
    if(clim == False):
        year_ax = ax.twiny()
        year_ax.set_frame_on(True)
        year_ax.patch.set_visible(False)
        year_ax.xaxis.set_ticks_position('bottom')
        year_ax.xaxis.set_label_position('bottom')
        year_ax.spines['bottom'].set_position(('outward', 30))
        year_ax.set_xticks(np.arange(1,len(timeaxis)+1, 12))
        year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
        year_ax.set_xlim(1, timeaxis[-1])
        

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

    #cbaxes = fig.add_axes([1.005, 0.075, 0.02, 0.885]) 
    #cbar1 = fig.colorbar(CF, cax=cbaxes)
    if(clim == False):
        cbar1 = fig.colorbar(CF, ax=[ax, year_ax], pad=0.015)
    else:
        cbar1 = fig.colorbar(CF, ax=ax, pad=0.015)
    #cbar1.set_label(colorunit, labelpad=4, y=0.5)
    cbar1.set_label(colorunit)

    #conf_int = 1.96*var_sd/np.sqrt(var_count)
    conf_int = var_sd
    conf_int[np.where(var_count < nmin)] = 1e5    
    conf_int[np.where(var_count == 0)] = np.nan    
    conf_int = ma.masked_invalid(conf_int)

    CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
    #fontP = FontProperties()
    #fontP.set_size('small')
    #legend([plot1], "title", prop=fontP)
    artists, labels = CF2.legend_elements(variable_name="\\sigma")
    labels[-1] = 'count $< $'+str(nmin)

    if(show_legend == True):
        if(clim == False):
            lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.19), fancybox=True, ncol=3)
        if(clim == True):
            lgd = plt.legend(artists, labels, handleheight=2, loc='upper left', bbox_to_anchor=(0.3, -0.1), fancybox=True, ncol=3)
    
    if(zmin != 0):
        ax.set_ylim(zmin, 0)
    else:
        ax.set_ylim(zlowest, 0)
    
    #plt.tight_layout();    
    if(save== True):
        if(show_legend == True):
            plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(savename, dpi=150)
    plt.show()


def find_sim_freshwater_h(lonmin, lonmax, latmin, latmax, plot=False, clim=False, year=0):
    if(clim == False):
        if(year == 0):
            print("\"year\" cannot be \"0\" when \"clim\" is \"False\"")
            return np.nan
        
        def find_lonlat_indices_seaice(seaice, time, lon, lat):
            time_ind = np.asscalar(np.argmin(np.abs(seaice.time - time )))
            return time_ind, np.asscalar(np.argmin(np.abs(seaice.lon - lon))), np.asscalar(np.argmin(np.abs(seaice.lat - lat)))

        seaice = xr.open_dataset("/media/data/Datasets/SeaIce/ocn_ana_2D_ll.nc")
        timeind, lonminind, latminind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmin, latmin)
        _, lonmaxind, latmaxind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmax, latmax)
        sim_monthly_area_average = np.zeros(12)
        sim_monthly_area_std = np.zeros(12)
        for i in range(12):
            sim_monthly_area_average[i] = np.nanmean(seaice.sim.isel(time=timeind+i, lon=slice(lonminind, lonmaxind), 
                                                          lat=slice(latminind, latmaxind)).values)
            sim_monthly_area_std[i] = np.nanstd(seaice.sim.isel(time=timeind+i, lon=slice(lonminind, lonmaxind), 
                                                          lat=slice(latminind, latmaxind)).values)

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

        fh = mass_diff / 1027. * 1e3

    if(clim == True):
        
        def find_lonlat_indices_seaice(seaice, time, lon, lat):
            time_ind = np.asscalar(np.argmin(np.abs(seaice.time - time )))
            return time_ind, np.asscalar(np.argmin(np.abs(seaice.lon - lon))), np.asscalar(np.argmin(np.abs(seaice.lat - lat)))

        seaice = xr.open_dataset("/media/data/Datasets/SeaIce/ocn_ana_2D_ll.nc")
        year = 2004
        _, lonminind, latminind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmin, latmin)
        _, lonmaxind, latmaxind = find_lonlat_indices_seaice(seaice, np.datetime64(str(year)+'-01'), lonmax, latmax)
        sim_monthly_area_average = np.zeros(12)
        sim_monthly_area_std = np.zeros(12)
        for i in range(12):
            monthmask = pd.to_datetime( pd.Series(seaice.time)).dt.month == i+1
            sim_monthly_area_average[i] = np.nanmean(seaice.sim.isel(time=monthmask, lon=slice(lonminind, lonmaxind), 
                                                          lat=slice(latminind, latmaxind)).values)
            sim_monthly_area_std[i] = np.nanstd(seaice.sim.isel(time=monthmask, lon=slice(lonminind, lonmaxind), 
                                                          lat=slice(latminind, latmaxind)).values)

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
        mass_diff_std = np.sqrt(sim_monthly_area_std[maxind]**2 + sim_monthly_area_std[minind]**2)
        fh = mass_diff / 1027. * 1e3
        fh_std = mass_diff_std / 1027. * 1e3
    
    return fh, fh_std







################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################


def conv180_360(lons):
    lons[lons < 0] = lons[lons < 0] + 360
    return lons

def conv360_180(lons):
    lons[lons > 180] = lons[lons > 180] - 360
    return lons

def mean_over_lons(dfg, lonmin, lonmax):
    return dfg.groupby_bins('lon', [lonmin, lonmax]).mean()

def mean_over_lons_alongTime(dfg, lonmin, lonmax):
    return dfg.groupby_bins('lon', [lonmin, lonmax]).mean(axis=1)


def compute_seaIceFlux_regional(seaIceFlux, latmin, latmax, lonmin, lonmax, timeStart=np.datetime64("2004-08-30"), 
                                timeEnd=np.datetime64("2008-08-30")):
    
    return seaIceFlux.net_ioflux.sel(time=slice(timeStart , timeEnd)).mean(axis=0).\
    groupby_bins('lat', [latmin, latmax]).apply(mean_over_lons, lonmin=lonmin, lonmax=lonmax)
    
def compute_seaIceFlux_regional_alongTime(latmin, latmax, lonmin, lonmax, timeStart=np.datetime64("2004-08-30"), 
                                timeEnd=np.datetime64("2008-08-30")):
    
    return seaIceFlux.net_ioflux.sel(time=slice(timeStart , timeEnd)).\
    groupby_bins('lat', [latmin, latmax]).apply(mean_over_lons_alongTime, lonmin=lonmin, lonmax=lonmax)


# provide lonmin, lonmax in 0 to 360 degrees longitudinal system
# Bathy, and dfmg are in -180 to +180 long system, make necessary conversions for them
def compute_freshwater_fluxes(df, PEclim, seaIceFlux, bathy, bins=5, wd=9, ht=6, varmin=33, varmax=35, nlevs=10,
                                    colorunit='Salinity (PSU)', save=False, savename="Untitled.png", 
                                    zbin=20, zmin=0, nmin=3, depth_max=0.0, levs=[], integrationDepth=100, plot=True, precip_dir="/media/data/Datasets/AirSeaFluxes/GPCPprecip",
                                    evap_dir="/media/data/Datasets/AirSeaFluxes/WHOIevap", clim=False, lonmin=np.nan, lonmax=np.nan, latmin=np.nan, latmax=np.nan, fontsize=8, show_legend=False):
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.close(1)
    fig = plt.figure(1, figsize=(wd, ht) )
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.02], height_ratios=[0.5,1, 0.25], hspace=0.3, wspace=0.08)
        
    evap_year_start, evap_year_end = 2004, 2015
    
    if(depth_max < 0):
        zlowest = depth_max
    else:
        zlowest = df.loc[:, 'DEPTH'].min()
    number_bins = np.abs(zlowest) // zbin
    depth_bins = np.linspace(zlowest, 0, number_bins)
        
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    timeaxis = np.arange(1, 13, 1)
    
    var_binned = np.zeros((len(timeaxis), len(depth_bins)))

    var_mean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalmean = np.zeros((len(timeaxis), len(depth_bins)-1))
    abssalstd = np.zeros((len(timeaxis), len(depth_bins)-1))
    rhomean = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_count = np.zeros((len(timeaxis), len(depth_bins)-1))
    var_sd = np.zeros((len(timeaxis), len(depth_bins)-1))
    freshwater_h = np.zeros(len(depth_bins)-1)
    freshwater_h_error = np.zeros(len(depth_bins)-1)

    no_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count = df.PSAL_ADJUSTED.count()

    PElats = PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).latitude.values
    PElons = PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).longitude.values    
    bathy_region = bathy.sel(lat=PElats, lon= conv360_180(PElons), method='nearest').elevation.values
    bathy_region_mask = (bathy_region < 0) & (bathy_region > -1000)
    
    if(count > 0):
        if(np.isnan(lonmin) & np.isnan(lonmax)):
            lonmin, lonmax = (df.loc[:, 'LONGITUDE'].min()), int(df.loc[:, 'LONGITUDE'].max())
            lonmin, lonmax = conv180_360([lonmin, lonmax])[0], conv180_360([lonmin, lonmax])[1]

        if not latmin:
            latmin, latmax = (df.loc[:, 'LATITUDE'].min()), int(df.loc[:, 'LATITUDE'].max())
        print(lonmin, lonmax, latmin, latmax)

        evap_total = (PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).where(bathy_region_mask).e.mean(dim=['latitude', 'longitude'])[:] * no_of_days[:]).sum()
        precip_total = (PEclim.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax)).where(bathy_region_mask).tp.mean(dim=['latitude', 'longitude'])[:] * no_of_days[:]).sum()
        seaIceFlux_total = compute_seaIceFlux_regional(seaIceFlux, latmin= latmin, latmax= latmax, lonmin= lonmin, lonmax= lonmax)
        runoff_2 = -(seaIceFlux_total + evap_total + precip_total)

        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            timeSlice_df = df.loc[monthmask]
            var_mean[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.mean().values
            abssalmean[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.mean().values
            abssalstd[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).SA.std().values
            rhomean[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).DENSITY_INSITU.mean().values
            var_count[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.count().values
            var_sd[j] = timeSlice_df.groupby(pd.cut(timeSlice_df.DEPTH, depth_bins)).PSAL_ADJUSTED.std().values

        for b in range(len(depth_bins)-1):

            if(np.isnan(var_mean[:, b]).all()):
                continue
            else:
                min_ind = np.nanargmin(abssalmean[:, b])
                max_ind = np.nanargmax(abssalmean[:, b])

                h_w = abs(depth_bins[1] - depth_bins[0])
                freshwater_h[b] = h_w * -(abssalmean[max_ind, b] - abssalmean[min_ind, b])

                freshwater_h_error[b] = abs(freshwater_h[b]) * np.sqrt( (abssalstd[:,b][max_ind] / abssalmean[:,b][max_ind])**2 +  (abssalstd[:,b][min_ind] / abssalmean[:,b][min_ind])**2)

    else:
        var_mean[i*12 : i*12+13] = np.nan

    gross_fh = np.nansum(freshwater_h * 1e-3)
    runoff_1 = gross_fh - (seaIceFlux_total + evap_total + precip_total)


    #var_mean = ma.masked_array(var_mean, mask= ma.masked_less(var_count, nmin).mask)
    #var_sd = ma.masked_array(var_sd, mask= ma.masked_less(var_count, nmin).mask)
    
    #fig.subplots_adjust(hspace=1.3)
    if(plot == True):
        ax = plt.subplot(gs[1,0])

        if(clim == False):
            year_ax = ax.twiny()
            year_ax.set_frame_on(True)
            year_ax.patch.set_visible(False)
            year_ax.xaxis.set_ticks_position('bottom')
            year_ax.xaxis.set_label_position('bottom')
            year_ax.spines['bottom'].set_position(('outward', 30))
            year_ax.set_xticks(np.arange(1,len(timeaxis)+1, 12))
            year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
            year_ax.set_xlim(1, timeaxis[-1])
            

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

        #cbaxes = fig.add_axes([1.005, 0.075, 0.02, 0.885]) 
        #cbar1 = fig.colorbar(CF, cax=cbaxes)
        colorbar_ax = plt.subplot(gs[1,1])
        
        cbar = Colorbar(ax = colorbar_ax, mappable = CF, orientation = 'vertical', extend='both', label= colorunit)

        #conf_int = 1.96*var_sd/np.sqrt(var_count)
        conf_int = var_sd
        conf_int[np.where(var_count < nmin)] = 1e5    
        conf_int[np.where(var_count == 0)] = np.nan    
        conf_int = ma.masked_invalid(conf_int)

        CF2 = ax.contourf(X.T[:, :], Y.T[:, :], conf_int[:, :], levels=[0, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf], colors='none', hatches=['', '/', '\\', '.', '+', 'o'])
        #fontP = FontProperties()
        #fontP.set_size('small')
        #legend([plot1], "title", prop=fontP)
        artists, labels = CF2.legend_elements(variable_name="\\sigma")
        labels[-1] = 'count $< $'+str(nmin)

        legend_ax = plt.subplot(gs[2,0], frameon=False)
        lgd = legend_ax.legend(artists, labels, handleheight=2, fancybox=True, ncol=3, loc=10)
        legend_ax.set_xticks([])
        legend_ax.set_xticklabels([])
        legend_ax.set_yticks([])
        legend_ax.set_yticklabels([])    

        if(zmin != 0):
            ax.set_ylim(zmin, 0)
        else:
            ax.set_ylim(zlowest, 0)

        fluxes_ax = plt.subplot(gs[0, 0])
        fnetbar = fluxes_ax.bar(1, gross_fh, width=0.2, label="$F_{net}$")
        enetbar = fluxes_ax.bar(2, evap_total, width=0.2, label="$E_{net}$")
        pnetbar = fluxes_ax.bar(3, precip_total, width=0.2, label="$P_{net}$")
        print("sea ice flux net", seaIceFlux_total.values[0][0])
        ficebar = fluxes_ax.bar(4, seaIceFlux_total.values[0][0], width=0.2, label="$F_{ice, net}$")
        runoffbar = fluxes_ax.bar(5, runoff_1.values[0][0], width=0.2, label="$R$")
        xticklabel = ['$F_{net}$', "$E_{net}$", "$P_{net}$", "$F_{ice, net}$",  "$R$"]
        fluxes_ax.set_xticks(np.arange(1,6))
        fluxes_ax.set_xticklabels(xticklabel, rotation=0)
        fluxes_ax.set_ylabel("$m\, yr^{-1}$")
        fluxes_ax.grid()
        
        if(save== True):
            if(show_legend == True):
                plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                plt.savefig(savename, dpi=150)
        plt.show()

    
    
    print(" CTD freshwater estimate = ",np.nansum(gross_fh), "\n runoff_2, -(E + Fice + P) = ", np.asscalar(runoff_2.values), "\n runoff_1,  F - (E + Fice + P) = ", np.asscalar(runoff_1.values) )
    return [[freshwater_h, freshwater_h_error] , \
             evap_total , precip_total, seaIceFlux_total , depth_bins, runoff_1, runoff_2]
