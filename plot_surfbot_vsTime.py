from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
import gsw
import matplotlib
from matplotlib import gridspec

# Function to plot the bottom properties
def plot_bottom_theta_sal_averages(ax, df, mask, title="", markersize=3, salmin=0, salmax=0, thetamin=-200, thetamax=-200, nmin=0, show_legend=False, fontsize=8, salticks=[], thetaticks=[], sampleplots=True, yearcounts=True, count_frame_on=False):
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements
        
    salmean = np.zeros(12)
    sal_sd = np.zeros(12)
    sal_count = np.zeros(12, dtype=int)
    sal_meop_error = np.zeros(12)
    sal_meop_error_sd = np.zeros(12)
    thetamean = np.zeros(12)
    theta_sd = np.zeros(12)
    theta_count = np.zeros(12, dtype=int)
    theta_meop_error = np.zeros(12)
    theta_meop_error_sd = np.zeros(12)
    salmonth_yearcount = np.zeros(12, dtype=int)
    thetamonth_yearcount = np.zeros(12, dtype=int)

    ## Standard deviation for bottom plots is outside the time bucket, all 12 values are the same and calculated for the entire year
    sal_sd[:] = df.loc[mask , 'PSAL_ADJUSTED'].std()
    theta_sd[:] = df.loc[mask , 'CTEMP'].std()
    sal_meop_error_sd = df.loc[mask, 'PSAL_ADJUSTED_ERROR'].std()     # 
    theta_meop_error_sd = df.loc[mask, 'TEMP_ADJUSTED_ERROR'].std()
    
    for i in range(12):
        monthmask = df['JULD'].dt.month == i+1  # selecting rows of data frame corresponding to the particular month
        sal_count[i] = df.loc[mask  & monthmask , 'PSAL_ADJUSTED'].count()
        theta_count[i] = df.loc[mask & monthmask, 'CTEMP'].count()
        
        salmean[i] = df.loc[mask  & monthmask, 'PSAL_ADJUSTED'].mean()
        sal_sd[i] = df.loc[mask  & monthmask, 'PSAL_ADJUSTED'].std()
        sal_meop_error[i] = df.loc[mask  & monthmask, 'PSAL_ADJUSTED_ERROR'].mean()
        
        thetamean[i] = df.loc[mask & monthmask, 'CTEMP'].mean()
        theta_sd[i] = df.loc[mask & monthmask, 'CTEMP'].std()
        # instrument error for temperature is not propogated to conserved temperature, instead, the temperature error is retained
        theta_meop_error[i] = df.loc[mask & monthmask, 'TEMP_ADJUSTED_ERROR'].mean() 

        salmonth_yearcount[i] = len(df.loc[mask & monthmask, 'JULD'].dt.year.unique())
        thetamonth_yearcount[i] = len(df.loc[mask & monthmask, 'JULD'].dt.year.unique())        
    ## The following masks data only if nmin is set to any value above 0
    ## salmean = ma.masked_array(salmean, mask = return_ma_nmin(sal_count, nmin) )
    ## sal_sd = ma.masked_array(sal_sd, mask = return_ma_nmin(sal_count, nmin) )
    ## thetamean = ma.masked_array(thetamean, mask = return_ma_nmin(theta_count, nmin))
    ## theta_sd = ma.masked_array(theta_sd, mask = return_ma_nmin(theta_count, nmin))
 
    # Using propogation of error and calculating standard error of mean using std(data+error)   
    sal_yerror = np.sqrt(sal_meop_error**2 + 1.96**2 * sal_sd**2 / sal_count )
    #sal_yerror = sal_sd
    theta_yerror = np.sqrt(theta_meop_error**2 + 1.96**2 * theta_sd**2 / theta_count )
    #theta_yerror = theta_sd
    
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    timeaxis = np.arange(1, 13, 1)
    #fig, ax = plt.subplots(figsize=(wd, ht))
    
    theta_ax = ax.twinx()
    theta_ax.errorbar(timeaxis, thetamean, yerr=theta_yerror, fmt='x', markersize=markersize, capsize=3, color='r', label="Pot. temp.")
    theta_ax.axhline(y = np.nanmean(thetamean) , color='r', linestyle='dashed', linewidth=1)
    ## if(np.any(theta_count) < 3):
    ##     indices_lt3 = np.where(theta_count < 3)[0]
    ##     theta_ax.errorbar(timeaxis[indices_lt3], thetamean[indices_lt3], yerr=theta_yerror[indices_lt3], fmt='x', markersize=markersize, capsize=3, color='k', label="Pot. temp.", zorder=1)
    
    theta_ax.set_xlim(0, timeaxis[-1]+1)
    theta_ax.set_xticks(timeaxis)
    theta_ax.set_xticklabels(timeaxis_ticklabel)
    theta_ax.set_ylabel('CT ($^o$C)', color='r')
    
    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=3, label="salinity", color='b')
    ax.axhline(y = np.nanmean(salmean), color='b', linestyle='dashed', linewidth=1)

    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    ax.set_ylabel('Salinity (PSU)', color='b')

    if not salticks:
        pass
    else:
        ax.set_yticks(salticks)
    if not thetaticks:
        pass
    else:
        theta_ax.set_yticks(thetaticks)
        
    if(salmin != 0 and salmax != 0):
        ax.set_ylim(salmin, salmax)
    if(thetamin != -200 and thetamax != -200):
        theta_ax.set_ylim(thetamin, thetamax)

    if(yearcounts == True):
        salcount_ax = ax.twiny()
        salcount_ax.set_xlim(0, timeaxis[-1]+1)        
        salcount_ax.set_xticks(timeaxis)
        salcount_ax.set_xticklabels(salmonth_yearcount)

    if(sampleplots == True):
        countax = ax.twinx()
        countax.set_frame_on(count_frame_on)
        countax.patch.set_visible(False)
        countax.yaxis.set_ticks_position('left')
        countax.yaxis.set_label_position('left')
        countax.spines['left'].set_position(('outward', 50))
        countax.bar(timeaxis, sal_count, 0.4, alpha=0.2, color='k')
        countax.set_yscale("log")
        countax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4])
        countax.set_ylim(1e-1, 5e4)
        countax.set_ylabel("Count (log scale)")
        if(count_frame_on == False):
            countax.set_axis_off()
        
    if(show_legend == True):
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]    
        handles2, labels2 = theta_ax.get_legend_handles_labels()
        handles2 = [h[0] for h in handles2]
        ax.legend(handles+handles2, labels+labels2, loc=0)
    
    ax.grid(alpha=0.5, linestyle='dotted')
    ax.set_title(title)
    return theta_ax, salcount_ax, countax

def compute_freshwater_input(h_w, abssalmean, rhomean):
    if(len(absalmean[np.isnan(abssalmean)]) > (len(abssalmean) - 2) ):
        return 0
    else:
        min_ind = np.nanargmin(abssalmean)
        max_ind = np.nanargmax(abssalmean)
        freshwater_h[i] = h_w * ( (rhomean[s:e][max_ind] * abssalmean[s:e][max_ind]) / (rhomean[s:e][min_ind] * abssalmean[s:e][min_ind])  - 1 ) * 1e3
        print(h_w, rhomean[s:e][max_ind], abssalmean[s:e][max_ind], rhomean[s:e][min_ind], abssalmean[s:e][min_ind])
        

def plot_surface_theta_sal_averages(ax, df, mask, title="", thetamin=-200, thetamax=-200, salmin=0, salmax=0, markersize=3, nmin=3, show_legend=True, fontsize=8, salticks=[], thetaticks=[], sampleplots=True, yearcounts=True, showplot=True, count_frame_on=True):
    matplotlib.rcParams.update({'font.size': fontsize})            
    salmean = np.zeros(12)
    sal_meop_error = np.zeros(12)
    sal_meop_error_sd = np.zeros(12)
    sal_sd = np.zeros(12)
    sal_count = np.zeros(12, dtype=int)
    thetamean = np.zeros(12)
    theta_meop_error = np.zeros(12)
    theta_meop_error_sd = np.zeros(12)
    theta_sd = np.zeros(12)
    theta_count = np.zeros(12, dtype=int)
    salmonth_yearcount = np.zeros(12, dtype=int)
    
    depthmask = df.DEPTH >= -100.0
    for i in range(12):
        monthmask = df['JULD'].dt.month == i+1
        sal_count[i] = df.loc[mask & monthmask & depthmask, 'PSAL_ADJUSTED'].count()
        theta_count[i] = df.loc[mask & monthmask & depthmask, 'CTEMP'].count()
        
        salmean[i] = df.loc[mask & monthmask & depthmask, 'PSAL_ADJUSTED'].mean()
        sal_meop_error[i] = df.loc[mask  & monthmask & depthmask, 'PSAL_ADJUSTED_ERROR'].mean()
        sal_meop_error_sd[i] = df.loc[mask & monthmask & depthmask, 'PSAL_ADJUSTED_ERROR'].std()
                
        thetamean[i] = df.loc[mask & monthmask & depthmask, 'CTEMP'].mean()
        theta_meop_error[i] = df.loc[mask  & monthmask & depthmask, 'TEMP_ADJUSTED_ERROR'].mean()
        theta_meop_error_sd[i] = df.loc[mask & monthmask & depthmask, 'TEMP_ADJUSTED_ERROR'].std()
        
        sal_sd[i] = df.loc[mask & monthmask & depthmask, 'PSAL_ADJUSTED'].std()
        theta_sd[i] = df.loc[mask & monthmask & depthmask, 'CTEMP'].std()
            
        salmonth_yearcount[i] = len(df.loc[mask & monthmask, 'JULD'].dt.year.unique())
    ## salmean = ma.masked_array(salmean, mask = return_ma_nmin(sal_count, nmin) )
    ## sal_sd = ma.masked_array(sal_sd, mask = return_ma_nmin(sal_count, nmin) )
    ## thetamean = ma.masked_array(thetamean, mask = return_ma_nmin(theta_count, nmin))
    ## theta_sd = ma.masked_array(theta_sd, mask = return_ma_nmin(theta_count, nmin))

    sal_yerror = np.sqrt(sal_meop_error**2 + 1.96**2 * sal_sd**2 /sal_count )
    #sal_yerror = sal_sd
    theta_yerror = np.sqrt(theta_meop_error**2 + 1.96**2 * theta_sd**2 / theta_count )
    #theta_yerror = theta_sd

    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
    timeaxis = np.arange(1, 13, 1)

    #fig, ax = plt.subplots(figsize=(wd, ht))
    
    theta_ax = ax.twinx()
    theta_ax.errorbar(timeaxis, thetamean, yerr=theta_yerror, fmt='x', markersize=markersize, capsize=3, color='r', label="Pot. temp.")
    theta_ax.axhline(y = np.nanmean(thetamean) , color='r', linestyle='dashed', linewidth=1)    
    theta_ax.set_xlim(0, timeaxis[-1]+1)
    theta_ax.set_xticks(timeaxis)
    theta_ax.set_xticklabels(timeaxis_ticklabel)
    theta_ax.set_ylabel("CT ($^o$C)", color='r')
    
    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=3, label="salinity", color='b')
    ax.axhline(y = np.nanmean(salmean), color='b', linestyle='dashed', linewidth=1)
    
    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    ax.set_ylabel('Salinity (PSU)', color='b')

    if not salticks:
        pass
    else:
        ax.set_yticks(salticks)
    if not thetaticks:
        pass
    else:
        theta_ax.set_yticks(thetaticks)

    if(salmin != 0 and salmax != 0):
        ax.set_ylim(salmin, salmax)
    if(thetamin != -200 and thetamax != -200):
        theta_ax.set_ylim(thetamin, thetamax)

    if(yearcounts == True):
        salcount_ax = ax.twiny()
        salcount_ax.set_xticks(np.arange(1,13,1))
        salcount_ax.set_xticklabels(salmonth_yearcount, rotation=0)
        salcount_ax.set_xlim(0, timeaxis[-1]+1)
    if(sampleplots == True):
        countax = ax.twinx()
        countax.set_frame_on(count_frame_on)
        countax.patch.set_visible(False)
        countax.yaxis.set_ticks_position('left')
        countax.yaxis.set_label_position('left')
        countax.spines['left'].set_position(('outward', 50))
        countax.bar(timeaxis, sal_count, 0.4, alpha=0.2, color='k')
        countax.set_yscale("log")
        countax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4])
        countax.set_ylim(1e-1, 5e4)
        countax.set_ylabel("Count (log scale)")
        
    if(show_legend == True):
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        handles2, labels2 = theta_ax.get_legend_handles_labels()
        handles2 = [h[0] for h in handles2]
        ax.legend(handles+handles2, labels+labels2, loc=2, framealpha=0.3)
            
    ax.grid(alpha=0.5, linestyle='dotted')
    ax.set_title(title)

    return theta_ax, salcount_ax, countax

    ## plt.tight_layout();
    ## if(save == True):
    ##     plt.savefig(savename, dpi=300)

    ## if(showplot == True):
    ##     plt.show()
    ## else:
    ##     plt.close()


### Ignore functions below ###############################################################################################################################################
def get_bottom_sal_averages_vs_year(df, mask, var='PSAL_ADJUSTED', years=[], save=False, savename="untitled.png", markersize=3, varmin=0, varmax=0, wd=10, ht=4, fontsize=14):
    matplotlib.rcParams.update({'font.size': fontsize})        
    if not years:
        years = np.sort(df.loc[mask, 'JULD'].dt.year.unique())
        
    salmean = np.zeros(len(years)*12)
    sal_sd = np.zeros(len(years)*12)
    sal_count = np.zeros(len(years)*12)
    for i in range(len(years)):
        yearmask = df['JULD'].dt.year == years[i]
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            salmean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, var].mean()

            sal_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, var].std()

            sal_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, var].count()

    sal_yerror = 1.96 * sal_sd / np.sqrt(sal_count)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * len(years)
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, len(years)*12+1, 1)
    fig, ax = plt.subplots(figsize=(wd, ht))
    year_ax = ax.twiny()
    fig.subplots_adjust(bottom=0.20)

    year_ax.set_frame_on(True)
    year_ax.patch.set_visible(False)
    year_ax.xaxis.set_ticks_position('bottom')
    year_ax.xaxis.set_label_position('bottom')
    year_ax.spines['bottom'].set_position(('outward', 30))

    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=2)
    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    if(varmin != 0 and varmax != 0):
        ax.set_ylim(varmin, varmax)
    ax.grid()

    #year_ax.plot(np.arange(len(years)), [34.6]*(len(years)), 'o', color='r')
    #year_ax.plot(timeaxis, [0]*len(timeaxis))
    year_ax.set_xticks(np.arange(1,len(timeaxis), 12))
    year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
    year_ax.set_xlim(0, timeaxis[-1]+1)

    #plt.ylim(salmin, salmax)

    if(save == True):
        plt.savefig(savename)
    plt.show()    


def get_surface_sal_averages_vs_year(df, mask, years=[], varmin=0, varmax=0, save=False, savename="untitled.png", markersize=3, var='PSAL_ADJUSTED', wd=10, ht=4):
    if not years:
        years = np.sort(df.loc[mask, 'JULD'].dt.year.unique())
        
    salmean = np.zeros(len(years)*12)
    sal_sd = np.zeros(len(years)*12)
    sal_count = np.zeros(len(years)*12)
    for i in range(len(years)):
        yearmask = df['JULD'].dt.year == years[i]
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            salmean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, var].mean()

            sal_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, var].std()

            sal_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, var].count()

    sal_yerror = 1.96 * sal_sd / np.sqrt(sal_count)

    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * len(years)
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, len(years)*12+1, 1)
    
    year_ax = ax.twiny()
    fig.subplots_adjust(bottom=0.20)

    year_ax.set_frame_on(True)
    year_ax.patch.set_visible(False)
    year_ax.xaxis.set_ticks_position('bottom')
    year_ax.xaxis.set_label_position('bottom')
    year_ax.spines['bottom'].set_position(('outward', 30))

    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=2)
    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    if(varmin != 0 and varmax != 0):
        ax.set_ylim(varmin, varmax)
    ax.grid()
    
    #year_ax.plot(np.arange(len(years)), [34.6]*(len(years)), 'o', color='r')
    #year_ax.plot(timeaxis, [0]*len(timeaxis))
    year_ax.set_xticks(np.arange(1,len(timeaxis), 12))
    year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
    year_ax.set_xlim(0, timeaxis[-1]+1)

    #plt.ylim(salmin, salmax)

    if(save == True):
        plt.savefig(savename)
    plt.show()    

def return_ma_nmin(arr_count, nmin):
    arr_count = ma.masked_less(arr_count, nmin)
    return arr_count.mask




def return_property_means(df, mask, var='CTEMP'):
    CTmean = df.loc[mask, var].mean()
    CTsd = df.loc[mask, var].std()
    if(var == 'CTEMP'):
        varError = 'TEMP_ADJUSTED_ERROR'
    else:
        varError = var+'_ERROR'
        
    CTerrorMean = df.loc[mask, varError].mean()
    CTcount = df.loc[mask, var].count()
    
    error = np.sqrt(CTerrorMean**2 + (1.96 * CTsd / np.sqrt(CTcount))**2)
    
    return CTmean , error



def plot_property_trends(df, ax, fontsize=8, save=False, savename="Untitled.png", markersize=3, thetamin=-2, thetamax=2, salmin=33.5, salmax=35, countmax=1e4, sigmamin=27, sigmamax=28,
                         hideYleft=False, hideYright=False, title=""):
    matplotlib.rcParams.update({'font.size': fontsize})            
    def get_years(df, years=[2010]):
        mask_year = [None] * len(years)
        for i in range(len(years)):
            mask_year[i] = df.loc[:, 'JULD'].dt.year == years[i]
        return mask_year
    
    yearly_count = df.groupby(df.JULD.dt.year)["CTEMP"].count().values
    
    years = df.loc[:, "JULD"].dt.year.unique()[yearly_count.nonzero()]
    #print(years)
    
    mask_years = get_years(df, years=years)
    
    sal_yearly_mean = np.zeros(len(years))
    salError_yearly_mean = np.zeros(len(years))
    CT_yearly_mean = np.zeros(len(years))
    CTError_yearly_mean = np.zeros(len(years))
    sigma0_yearly_mean = np.zeros(len(years))
    sigma0Error_yearly_mean = np.zeros(len(years))
    
    for i in range(len(years)):
        try:
            sal_yearly_mean[i], salError_yearly_mean[i] =  return_property_means(df, mask_years[i], var="PSAL_ADJUSTED")
            CT_yearly_mean[i], CTError_yearly_mean[i] = return_property_means(df, mask_years[i], var="CTEMP")
            sigma0_yearly_mean[i] = df.loc[mask_years[i], "POT_DENSITY"].mean()
            sigma0Error_yearly_mean[i] = df.loc[mask_years[i], "POT_DENSITY"].std()
        except:
            pass
        
    ax.errorbar(years[~np.isnan(sal_yearly_mean)], sal_yearly_mean[~np.isnan(sal_yearly_mean)], yerr= salError_yearly_mean[~np.isnan(sal_yearly_mean)], fmt='o', markersize=markersize, capsize=3, label="salinity", color='b')
                
    z, res, _, _, _ = np.polyfit(years[~np.isnan(sal_yearly_mean)], sal_yearly_mean[~np.isnan(sal_yearly_mean)],
                                 1, full=True)
    p = np.poly1d(z)
    
    ax.plot(years[~np.isnan(sal_yearly_mean)], p(years[~np.isnan(sal_yearly_mean)]),"b", linestyle="--")
    ax.set_xlabel("Years")

    #ax.set_xticks(ax.get_xticks, rotation=45)
    ax.set_title(title)
    ax.set_ylim(salmin, salmax)
    if hideYleft:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("PSU", color="b")
        
    theta_ax = ax.twinx()
    theta_ax.errorbar(years[~np.isnan(CT_yearly_mean)], CT_yearly_mean[~np.isnan(sal_yearly_mean)], yerr= CTError_yearly_mean[~np.isnan(CT_yearly_mean)], fmt='x', markersize=markersize, capsize=3, label="CT", color='r')                
    z, res, _, _, _ = np.polyfit(years[~np.isnan(CT_yearly_mean)], CT_yearly_mean[~np.isnan(CT_yearly_mean)],
                                 1, full=True)
    p = np.poly1d(z)    
    theta_ax.plot(years[~np.isnan(CT_yearly_mean)], p(years[~np.isnan(CT_yearly_mean)]),"r--")
    
    theta_ax.set_ylim(thetamin, thetamax)
    if hideYright:
        theta_ax.set_yticklabels([])
    else:
        theta_ax.set_ylabel("CT", color="r")        

    sigma0_ax = ax.twinx()
    sigma0_ax.errorbar(years[~np.isnan(sigma0_yearly_mean)], sigma0_yearly_mean[~np.isnan(sigma0_yearly_mean)], yerr= sigma0Error_yearly_mean[~np.isnan(sigma0_yearly_mean)], fmt='^', markersize=markersize, capsize=3, label="$\sigma_O$", color='k')                
    z, res, _, _, _ = np.polyfit(years[~np.isnan(sigma0_yearly_mean)], sigma0_yearly_mean[~np.isnan(sigma0_yearly_mean)],
                                 1, full=True)
    p = np.poly1d(z)    
    sigma0_ax.plot(years[~np.isnan(sigma0_yearly_mean)], p(years[~np.isnan(sigma0_yearly_mean)]),"k--")

    sigma0_ax.patch.set_visible(False)
    sigma0_ax.yaxis.set_ticks_position('left')
    sigma0_ax.yaxis.set_label_position('left')
    sigma0_ax.spines['left'].set_position(('outward', 50))
    sigma0_ax.set_ylim(sigmamin, sigmamax)
    if hideYleft:
        sigma0_ax.set_yticklabels([])
        sigma0_ax.set_axis_off()        
    else:
        sigma0_ax.set_ylabel("$\sigma_O$ (kgm$^{-3}$)", color="k")        

    countax = ax.twinx()
    #countax.set_frame_on(False)
    countax.patch.set_visible(False)
    countax.yaxis.set_ticks_position('left')
    countax.yaxis.set_label_position('left')
    countax.spines['left'].set_position(('outward', 100))
    countax.bar(years[~np.isnan(sal_yearly_mean)], 
                (yearly_count[yearly_count.nonzero()])[~np.isnan(sal_yearly_mean)], 0.4, alpha=0.2, color='k')
    countax.set_yscale("log")
    countax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4])
    countax.set_ylim(1e-1, 5e4)

    countax.set_ylim(0, 1e4)
    if hideYleft:
        countax.set_yticklabels([])
        countax.set_axis_off()        
    else:
        countax.set_ylabel("Count (log scale)")        


    ax.set_xticklabels( np.array(ax.get_xticks(), dtype=int ) )            
    if(save==True):
        plt.savefig(savename)
    #plt.show()
    #print("y=%.6fx+(%.6f)"%(z[0],z[1]), "res=",res)
    #return sal_yearly_mean, sd, n



def plot_properties_zonally(df, ax, long_bins=np.arange(0, 360.1, 5), markersize=3, show_legend=True, salmin=0, salmax=0, thetamin=-200, thetamax=-200, sigma_min=-200, sigma_max=-200, hideYleft=False, hideYright=False, sigma_frame_on=True):
        
    sal_error_long_bins = df["PSAL_ADJUSTED_ERROR"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).mean()
    sal_std_long_bins = df["PSAL_ADJUSTED"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).std()
    sal_count_long_bins = df["PSAL_ADJUSTED"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).count()
    sal_stat_error_long_bins = 1.96 * sal_std_long_bins / np.sqrt(sal_count_long_bins)
    sal_error = np.sqrt(sal_error_long_bins**2 + sal_stat_error_long_bins**2)
    sal_long_bins = df["PSAL_ADJUSTED"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).mean().values
    
    CT_error_long_bins = df["TEMP_ADJUSTED_ERROR"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).mean()
    CT_std_long_bins = df["CTEMP"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).std()
    CT_count_long_bins = df["CTEMP"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).count()
    CT_stat_error_long_bins = 1.96 * CT_std_long_bins / np.sqrt(CT_count_long_bins)
    CT_error = np.sqrt(CT_error_long_bins**2 + CT_stat_error_long_bins**2)
    CT_long_bins = df["CTEMP"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).mean().values
    
    sigma_std_long_bins = df["POT_DENSITY"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).std()
    sigma_count_long_bins = df["POT_DENSITY"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).count()
    sigma_error = 1.96 * sigma_std_long_bins / np.sqrt(sigma_count_long_bins)
    sigma_long_bins = df["POT_DENSITY"].groupby(pd.cut(df.LONGITUDE, long_bins ) ).mean().values

    def yearly_count(gdf):
        yearly_count = gdf.groupby(gdf.JULD.dt.year).CTEMP.count()
        return len(yearly_count[yearly_count > 10])

    lonbin_count = df.groupby(pd.cut(df.LONGITUDE, long_bins ) )["CTEMP"].count().values
    lonbin_yearly_count = df.groupby(pd.cut(df.LONGITUDE, long_bins ) ).apply(yearly_count)
    
    ax.errorbar(long_bins[1:], sal_long_bins, yerr=sal_error, fmt='o', markersize=markersize, 
                capsize=3, color='b', label="Salinity")
    ax_right = ax.twinx()
    ax_right.errorbar(long_bins[1:], CT_long_bins, yerr=CT_error, fmt='x', markersize=markersize, 
                      capsize=3, color='r', label="Pot. temp.")

    sigma_ax = ax.twinx()
    sigma_ax.set_frame_on(sigma_frame_on)
    sigma_ax.patch.set_visible(False)
    sigma_ax.yaxis.set_ticks_position('left')
    sigma_ax.yaxis.set_label_position('left')
    sigma_ax.spines['left'].set_position(('outward', 50))
    sigma_ax.errorbar(long_bins[1:], sigma_long_bins, yerr=sigma_error, fmt='+', markersize=markersize, 
                      capsize=3, color='0.5', label="Pot. density")
    
    sigma_ax.axhline(28, color='k', linestyle="--")
    sigma_ax.axhline(28.27, color='k', linestyle="--")
    sigma_ax.axhline(28.35, color='k', linestyle="--")
    
    #sigma_ax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4])
    #sigma_ax.set_ylim(1e-1, 5e4)
    sigma_ax.set_ylabel("Pot. density")
    if(sigma_frame_on == False):
        sigma_ax.set_axis_off()
    

    if(salmin != 0 and salmax != 0):
        ax.set_ylim(salmin, salmax)
    if(thetamin != -200 and thetamax != -200):
        ax_right.set_ylim(thetamin, thetamax)
    if(sigma_min != -200 and sigma_max != -200):
        sigma_ax.set_ylim(sigma_min, sigma_max)
    
    if(show_legend == True):
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]    
        handles2, labels2 = ax_right.get_legend_handles_labels()
        handles2 = [h[0] for h in handles2]
        handles3, labels3 = sigma_ax.get_legend_handles_labels()
        handles3 = [h[0] for h in handles3]
        
        ax.legend(handles+handles2+handles3, labels+labels2+labels3, loc=0)
        
    countax = ax.twinx()
    #countax.set_frame_on(False)
    countax.patch.set_visible(False)
    countax.yaxis.set_ticks_position('left')
    countax.yaxis.set_label_position('left')
    countax.spines['left'].set_position(('outward', 100))
    countax.bar(long_bins[1:], 
                lonbin_count, 5, alpha=0.2, color='k')
    countax.set_yscale("log")
    countax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4])
    countax.set_ylim(1e-1, 5e4)

    countax.set_ylim(0, 1e4)
    if hideYleft:
        countax.set_yticklabels([])
        countax.set_axis_off()        
    else:
        countax.set_ylabel("Count (log scale)")        
    
    
    ax.set_xlim(0, 360)
    ax.grid()
    plt.show()
