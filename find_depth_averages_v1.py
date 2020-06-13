from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import scipy.interpolate as interpolate
import pdb
import numpy.ma as ma

def find_depth_averages(df, wd=7, ht=5, tmin = -2.3, tmax=-0.5, salmin=33.5, salmax=35.4, 
                        sigmin=26, sigmax=28, sigvline=[0,0], save=False, savename='Untitled.png', 
                        depth_increment=-10.0, mask_depth_step=-5.0, use_sample_sd=False, use_full_sd=True, 
                        cutoff_depth=0, errorevery=1, ymin=0, maskerror=0, markersize='4'):
    zlowest = df['DEPTH'].min()
    zhighest = 0.0
    z = np.arange(zhighest, zlowest, depth_increment)
    depth_bins = np.arange(zlowest, zhighest, -depth_increment)
    sal_z = np.zeros(len(z))
    count_sal = np.zeros(len(z))
    temp_z = np.zeros(len(z))
    count_temp = np.zeros(len(z))
    sigma_z = np.zeros(len(z))
    count_sigma = np.zeros(len(z))
    
    def get_full_sample_sd(var):
        SD = np.zeros(len(z))
        N_SD = np.zeros(len(z))
        for i in range(len(z)):
            mds = mask_depth_step
            flag = 0
            if(use_full_sd == True):
                mask_depth_full = (dfmg['DEPTH'] < (z[i] - mds)) & (dfmg['DEPTH'] > 
                                                                            (z[i] + mds))
                SD[i] = dfmg.loc[mask_depth_full, var].std()
            if(use_sample_sd == True):
                mask_depth_sample = (df['DEPTH'] < (z[i] - mds)) & (df['DEPTH'] > 
                                                                            (z[i] + mds))
                SD[i] = df.loc[mask_depth_sample, var].std()
                
            N_SD[i] = len(df.loc[mask_depth_sample, var].dropna())
            
            while(N_SD[i] == 0 & flag == 0):
                if(use_full_sd == True):
                    mask_depth_full = (dfmg['DEPTH'] < (z[i] - mds)) & (dfmg['DEPTH'] > 
                                                                            (z[i] + mds))
                    SD[i] = dfmg.loc[mask_depth_full, var].std()
                if(use_sample_sd == True):
                    mask_depth_sample = (df['DEPTH'] < (z[i] - mds)) & (df['DEPTH'] > 
                                                                            (z[i] + mds))
                    SD[i] = df.loc[mask_depth_sample, var].std()
                
                N_SD[i] = len(df.loc[mask_depth_sample, var].dropna())
                if(N_SD[i] == 0):
                    mds = mds-5.0
                    flag = 1
                    
            sd_max = max(SD)
            SD[np.isnan(SD)] = sd_max
            N_SD[N_SD == 0] = 1
        return SD, N_SD
    
    def get_var_to_nearest_z_level(dfprof):
        sal = dfprof['PSAL_ADJUSTED'].dropna().values
        depth_sal = dfprof.loc[dfprof['PSAL_ADJUSTED'].dropna().index, 'DEPTH'].values

        temp = dfprof['CTEMP'].dropna().values
        depth_temp = dfprof.loc[dfprof['CTEMP'].dropna().index, 'DEPTH'].values

        sigma = dfprof['POT_DENSITY'].dropna().values
        depth_sigma = dfprof.loc[dfprof['POT_DENSITY'].dropna().index, 'DEPTH'].values
        
        if((len(sal) > 1) & (len(temp) > 1) & (len(sigma) > 1)):
            def get_z_range_ind(depth):
                z_start = np.argmin(abs(z - depth[0]))
                z_end = np.argmin(abs(z - depth[-1]))
                return z_start, z_end

            def get_interpd_values(var, depth):
                interp_func = interpolate.interp1d(depth, var)
                return interp_func(z[z_start+1:z_end])

            z_start, z_end = get_z_range_ind(depth_sal)
            sal_z[z_start+1:z_end] += get_interpd_values(sal[:], depth_sal[:])
            count_sal[z_start+1:z_end] += 1

            #pdb.set_trace()
            z_start, z_end = get_z_range_ind(depth_temp)
            temp_z[z_start+1:z_end] += get_interpd_values(temp[:], depth_temp[:])
            count_temp[z_start+1: z_end] += 1

            z_start, z_end = get_z_range_ind(depth_sigma)
            sigma_z[z_start+1:z_end] += get_interpd_values(sigma[:], depth_sigma[:])
            count_sigma[z_start+1: z_end] += 1
        
    df.groupby('PROFILE_NUMBER').apply(get_var_to_nearest_z_level)
    
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
    
    sd, n_sd = get_full_sample_sd('CTEMP')
    xerror = get_xerror(sd, n_sd)
    avg = temp_z[1:-1]/count_temp[1:-1]
    axT.errorbar(avg[~xerror.mask[1:-1]], z[1:-1][~xerror.mask[1:-1]], xerr=xerror[1:-1][~xerror.mask[1:-1]], 
                 errorevery=errorevery, capsize=4, 
                 color='b', fmt='.', markersize=markersize)
    
    axT.set_xlabel("$\\theta^o$ C", color='b')
    axT.set_xlim(tmin, tmax)
    if(cutoff_depth != 0):
        axT.set_ylim(cutoff_depth, 0)
    axT.set_ylabel('Depth (m)')
    
    sd, n_sd = get_full_sample_sd('PSAL_ADJUSTED')
    xerror = get_xerror(sd, n_sd)
    avg = sal_z[1:-1]/count_sal[1:-1]
    axS.errorbar(avg[~xerror.mask[1:-1]], z[1:-1][~xerror.mask[1:-1]], xerr=xerror[1:-1][~xerror.mask[1:-1]], 
                 errorevery=errorevery, capsize=4, color='r', fmt='.', markersize=markersize)
    axS.set_xlim(salmin, salmax)
    axS.set_xlabel("Salinity", color='r')
    
    sd, n_sd = get_full_sample_sd('POT_DENSITY')
    xerror = get_xerror(sd, n_sd)
    avg = sigma_z[1:-1]/count_sigma[1:-1]
    axSig.errorbar(avg[~xerror.mask[1:-1]], z[1:-1][~xerror.mask[1:-1]], xerr=xerror[1:-1][~xerror.mask[1:-1]], 
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
