import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma


def find_lonlat_indices_evap(evap_year, lon, lat):
    if(lon >=0 & lon <= 180):
        longrid = evap_year.lon.values
    else:
        longrid = evap_year.lon.values
        longrid[180:360] = evap_year.lon.values[180:360] - 360
    
    return np.asscalar(np.argmin(np.abs(longrid - lon))), np.asscalar(np.argmin(np.abs(evap_year.lat - lat)))

def find_lonlat_indices_precip(precip, time, lon, lat):
    if(lon >=0 & lon <= 180):
        longrid = precip.lon.values
    else:
        longrid = precip.lon.values
        mid_ind = np.asscalar(np.argmin(np.abs(precip.lon - 180.5) ))
        longrid[mid_ind : -1] = precip.lon.values[mid_ind:-1] - 360
    time_ind = np.asscalar(np.argmin(np.abs(precip.time - time )))
    return time_ind, np.asscalar(np.argmin(np.abs(longrid - lon))), np.asscalar(np.argmin(np.abs(precip.lat - lat)))

def freshwater_flux_compute(evap, precip, precip_error, year, lonmin, lonmax, latmin, latmax, wd=7, ht=4, plot=True, save=False, savename="untitled.png"):
    from calendar import monthrange
    
    elonminind , elatminind = find_lonlat_indices_evap(evap, lonmin, latmin)
    elonmaxind, elatmaxind = find_lonlat_indices_evap(evap, lonmax, latmax)
    
    ptimeminind, plonminind, platminind = find_lonlat_indices_precip(precip, np.datetime64(str(year)+"-01-01"), 
                                                                     lonmin, latmin)
    ptimemaxind, plonmaxind, platmaxind = find_lonlat_indices_precip(precip, np.datetime64(str(year)+"-12-01"), 
                                                                     lonmax, latmax)
    #print(elonminind, elonmaxind, plonminind, plonmaxind)
    E = evap.evapr.isel(lon=slice(elonminind, elonmaxind), lat=slice(elatminind, elatmaxind)).values*10
    E_err = evap.err.isel(lon=slice(elonminind, elonmaxind), lat=slice(elatminind, elatmaxind)).values*10
    
    P = precip.precip.isel(time=slice(ptimeminind, ptimemaxind+1), lon=slice(plonminind, plonmaxind), 
                          lat=slice(platminind, platmaxind)).values
    P_err = precip_error.precip.isel(time=slice(ptimeminind, ptimemaxind+1), lon=slice(plonminind, plonmaxind), 
                          lat=slice(platminind, platmaxind)).values
    
    E_mon_mean, P_mon_mean = np.zeros(12), np.zeros(12)
    E_mon_mean_err, P_mon_mean_err = np.zeros(12), np.zeros(12)
    
    for i in range(12):
        E_mon_mean[i] = np.nanmean(E[i]) * float(monthrange(year, i+1)[1]) / 365.25 # to convert cm/yr to mm/month
        E_mon_mean_err[i] = np.nanmean(E_err[i]) * float(monthrange(year, i+1)[1]) / 365.25 # to convert cm/yr to mm/month
        
        P_mon_mean[i] = np.nanmean(P[i]) * float(monthrange(year, i+1)[1])
        P_mon_mean_err[i] = np.nanmean(P_err[i]) * float(monthrange(year, i+1)[1])
        
    E_mon_mean[np.isnan(E_mon_mean)] = 0.0
    E_mon_mean_err[np.isnan(E_mon_mean_err)] = 0.0
    
    P_mon_mean[np.isnan(P_mon_mean)] = 0.0
    P_mon_mean_err[np.isnan(P_mon_mean_err)] = 0.0
    
    if(plot == True):
        fig, ax = plt.subplots(figsize=(wd, ht))
        ax.errorbar(np.arange(1,13,1), E_mon_mean - P_mon_mean, yerr=(E_mon_mean_err+P_mon_mean_err), capsize=3, fmt='o', 
                    markersize=5)
        ax.set_xticks(np.arange(1,13,1))
        ax.set_xticklabels(month_names)
        if(save == True):
            plt.savefig(savename)
        plt.show()
    return ( np.nansum(E_mon_mean) - np.nansum(P_mon_mean) ) , max(E_mon_mean_err+P_mon_mean_err)
