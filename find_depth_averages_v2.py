from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma

def find_depth_averages_v2(df, wd=7, ht=5, tmin = -2.3, tmax=-0.5, salmin=33.5, salmax=35.4,
                           sigmin=26, sigmax=28, sigvline=[0,0], save=False, savename='Untitled.png',
                           depth_bins=10.0, mask_depth_step=-5.0, use_sample_sd=False, use_full_sd=True,
                           cutoff_depth=0, errorevery=1, ymin=0, maskerror=0, markersize='4', min_n=25):
    zlowest = df['DEPTH'].min()
    zhighest = 0.0
    
    depth_binned = np.arange(zlowest, zhighest, depth_bins)
        
    binned_groups = df.groupby(pd.cut(df.DEPTH, depth_binned))
    
    sal_mean = binned_groups['PSAL_ADJUSTED'].mean().values
    sal_count = binned_groups['PSAL_ADJUSTED'].count().values
    sal_sd = binned_groups['PSAL_ADJUSTED'].std().values
    
    temp_mean = binned_groups['TEMP_ADJUSTED'].mean().values
    temp_count = binned_groups['TEMP_ADJUSTED'].count().values
    temp_sd = binned_groups['TEMP_ADJUSTED'].std().values

    sigma_mean = binned_groups['POT_DENSITY'].mean().values
    sigma_count = binned_groups['POT_DENSITY'].count().values
    sigma_sd = binned_groups['POT_DENSITY'].std().values

    
    def get_xerror(sd, n_sd):
        xerror = 1.96 * sd / np.sqrt(n_sd)
        if(maskerror > 0):
            xerror = ma.array(xerror, mask=(xerror > np.percentile(xerror, maskerror) ))
        else:
            xerror = ma.array(xerror, mask=(xerror < 0))
        return xerror
    
    fig = plt.figure(figsize=(wd,ht))
    axT = host_subplot(1,1,1, axes_class=AA.Axes)
    axS = axT.twiny()
    axSig = axT.twiny()
    
    offset = 40
    new_fixed_axis = axSig.get_grid_helper().new_fixed_axis
    axSig.axis["top"] = new_fixed_axis(loc="top",
                                    axes=axSig,
                                    offset=(0,offset))
    axSig.axis["top"].toggle(all=True)
    temp_xerror = get_xerror(temp_sd, temp_count)
    mask_low = np.where(temp_count > min_n)[0]
    
    axT.errorbar(temp_mean[mask_low], depth_binned[:-1][mask_low], xerr=temp_xerror[mask_low], 
                 errorevery=errorevery, capsize=4, 
                 color='b', fmt='.', markersize=markersize)
    
    axT.set_xlabel("$\\theta^o$ C", color='b')
    axT.set_xlim(tmin, tmax)
    if(cutoff_depth != 0):
        axT.set_ylim(cutoff_depth, 0)
    axT.set_ylabel('Depth (m)')
    
    sal_xerror = get_xerror(sal_sd, sal_count)
    mask_low = np.where(sal_count > min_n)[0]
    axS.errorbar(sal_mean[mask_low], depth_binned[:-1][mask_low], xerr=sal_xerror[mask_low], 
                 errorevery=errorevery, capsize=4, color='r', fmt='.', markersize=markersize)
    axS.set_xlim(salmin, salmax)
    axS.set_xlabel("Salinity", color='r')
    
    sigma_xerror = get_xerror(sigma_sd, sigma_count)
    mask_low = np.where(sigma_count > min_n)[0]
    axSig.errorbar(sigma_mean[mask_low], depth_binned[:-1][mask_low], xerr=sigma_xerror[mask_low], 
                   errorevery=errorevery, 
                   capsize=4, color='g', fmt='.', markersize=markersize)
    axSig.set_xlabel('$\sigma_0$(kgm$^{-3}$)', color='g')
    axSig.set_xlim(sigmin, sigmax)
    axSig.axis["top"].label.set_color('g')
    axSig.locator_params(axis='x',nticks=4)
    axSig.axvline(x = sigvline[0], color='g', linestyle='--')
    axSig.axvline(x = sigvline[1], color='g', linestyle='--')
    
    plt.tight_layout();
    if(save==True):
        plt.savefig(savename, dpi=150)
    
    plt.show();
