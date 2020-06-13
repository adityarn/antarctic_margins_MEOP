def plot_var_contours_with_distance(df, mask, var, dist=100, bins=5, wd=12, ht=5, varmin=33, varmax=35, nlevs=10,
                                    colorunit=' ', save=False, savename="Untitled.png", 
                                    zbin=10, xbin=10, zmin=0, nmin=0):
    zlowest = df.loc[mask, 'DEPTH'].min()
    depth_bins = np.arange(zlowest, 0+zbin, zbin)
    dist_bins = np.arange(0, dist+xbin, xbin)
    var_binned = np.zeros((len(dist_bins), len(depth_bins)))
    
    dist_binned_group = df.loc[mask].groupby(pd.cut(df.loc[mask].DIST_GLINE, dist_bins))
    var_mean = np.zeros((len(dist_bins), len(depth_bins)-1))
    var_count = np.zeros((len(dist_bins), len(depth_bins)-1))
    var_sd = np.zeros((len(dist_bins), len(depth_bins)-1))
    i = 0
    for groupList, xGroup in dist_binned_group:
        zGroup = xGroup.groupby(pd.cut(xGroup.DEPTH, depth_bins))
        var_mean[i] = zGroup[var].mean().values
        var_count[i] = zGroup[var].count().values
        var_sd[i] = zGroup[var].std().values
        i += 1

    var_mean = ma.masked_array(var_mean)
    var_mean = ma.masked_less(var_count, nmin)
    fig, ax = plt.subplots(1,2, figsize=(wd, ht))
    #fig.subplots_adjust(hspace=1.3)
    X, Y = np.meshgrid(dist_bins[:], depth_bins[:-1])
    levels = np.linspace(varmin, varmax, nlevs)
    CF = ax[0].contourf(X.T[:-1, :], Y.T[:-1, :], var_mean[:-1, :], levels)
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('Distance from grounding line (km)')
    if(zmin != 0):
        ax[0].set_ylim(zmin, 0)
    else:
        ax[0].set_ylim(zlowest, 0)
    cbar1 = fig.colorbar(CF, ax=ax[0])
    cbar1.set_label(colorunit)
    
    conf_int = 1.96*var_sd/np.sqrt(var_count)
    conf_int[np.isnan(conf_int)] = 100000.0
    levels2 = np.linspace(0,1, 10)
    print(np.max(conf_int[~np.isnan(conf_int)]))
    CF2 = ax[1].contourf(X.T[:-1, :], Y.T[:-1, :], conf_int[:-1, :], levels2)
    ax[1].set_xlabel('Distance from grounding line (km)')
    if(zmin != 0):
        ax[1].set_ylim(zmin, 0)
    cbar2 = fig.colorbar(CF2, ax=ax[1])
    cbar2.set_label('Error in sample mean')
    if(save== True):
        plt.savefig(savename)
    plt.show()

def plot_dist_quarters_region(df, mask_region, varname='CTEMP', varunit='$\\theta^o$C', varmin=-2, 
                              varmax=1.6, zmin=0, savename='Untitled', nlevs=10, dist=160):
    months_quarters = np.array([[12,1,2], [3,4,5], [6,7,8], [9,10,11]])
    mth_qt_names = ['DJF', 'MAM', 'JJA', 'SON']

    for i in range(4):
        mask_months = sel_months(df, months_quarters[i])
        mask = get_mask_from_prof_mask(df, (mask_region & mask_months ) )
        plot_var_contours_with_distance(df, mask=mask, var=varname, dist=dist, 
                                        colorunit=varunit, save=True, 
                                        savename=savename+mth_qt_names[i]+'.png', 
                                        wd=14, ht=5, 
                                        zmin=zmin, varmin=varmin, varmax=varmax, nlevs=nlevs)
