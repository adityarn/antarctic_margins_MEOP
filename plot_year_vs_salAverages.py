from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
import gsw

def get_bottom_sal_averages_vs_year(df, mask, var='PSAL_ADJUSTED', years=[], save=False, savename="untitled.png", markersize=3, varmin=0, varmax=0, wd=10, ht=4):
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

def return_ma_nmin(arr_count, nmin):
    arr_count = ma.masked_less(arr_count, nmin)
    return arr_count.mask

def get_bottom_theta_sal_averages_vs_year(df, mask, years=[], save=False, savename="untitled.png", markersize=3, salmin=0, salmax=0, thetamin=0, thetamax=0, wd=10, ht=4, nmin=3, clim=False):
    if not years:
        years = np.sort(df.loc[mask, 'JULD'].dt.year.unique())
    if(clim == False):
        iter_range = len(years)
    else:
        iter_range = 1
        
    salmean = np.zeros(iter_range*12)
    sal_sd = np.zeros(iter_range*12)
    sal_count = np.zeros(iter_range*12)
    thetamean = np.zeros(iter_range*12)
    theta_sd = np.zeros(iter_range*12)
    theta_count = np.zeros(iter_range*12)

    for i in range(iter_range):
        if(clim == False):
            yearmask = df['JULD'].dt.year == years[i]
        if(clim == True):
            yearmask = df['JULD'].dt.year.isin(years)
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            salmean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'PSAL_ADJUSTED'].mean()
            thetamean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'CTEMP'].mean()

            sal_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'PSAL_ADJUSTED'].std()
            theta_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'CTEMP'].std()
            
            sal_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'PSAL_ADJUSTED'].count()
            theta_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').tail(1).index, 'CTEMP'].count()

    salmean = ma.masked_array(salmean, mask = return_ma_nmin(sal_count, nmin) )
    sal_sd = ma.masked_array(sal_sd, mask = return_ma_nmin(sal_count, nmin) )
    thetamean = ma.masked_array(thetamean, mask = return_ma_nmin(theta_count, nmin))
    theta_sd = ma.masked_array(theta_sd, mask = return_ma_nmin(theta_count, nmin))
    
    #sal_yerror = 1.96 * sal_sd / np.sqrt(sal_count)
    sal_yerror = sal_sd
    #theta_yerror = 1.96 * theta_sd / np.sqrt(theta_count)
    theta_yerror = theta_sd
    
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * iter_range
    timeaxis_yearticklabel = years
    timeaxis = np.arange(1, iter_range*12+1, 1)
    fig, ax = plt.subplots(figsize=(wd, ht))
    
    theta_ax = ax.twinx()

    if(clim == False):
        year_ax = ax.twiny()
        fig.subplots_adjust(bottom=0.20)

        year_ax.set_frame_on(True)
        year_ax.patch.set_visible(False)
        year_ax.xaxis.set_ticks_position('bottom')
        year_ax.xaxis.set_label_position('bottom')
        year_ax.spines['bottom'].set_position(('outward', 30))
        year_ax.set_xticks(np.arange(1,len(timeaxis), 12))
        year_ax.set_xticklabels(np.array(years, dtype=str), rotation='0')
        year_ax.set_xlim(0, timeaxis[-1]+1)
        

    theta_ax.errorbar(timeaxis, thetamean, yerr=theta_yerror, fmt='x', markersize=markersize, capsize=3, color='r', label="Pot. temp.", zorder=1)
    theta_ax.set_xlim(0, timeaxis[-1]+1)
    theta_ax.set_xticks(timeaxis)
    theta_ax.set_xticklabels(timeaxis_ticklabel)
    theta_ax.set_ylabel('Pot. temp. ($\\theta^o$C)')
    
    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=3, label="salinity", color='b', zorder=3)
    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    ax.set_ylabel('Salinity (Cond.)')
    
    if(salmin != 0 and salmax != 0):
        ax.set_ylim(salmin, salmax)
    if(thetamin != 0 or thetamax != 0):
        theta_ax.set_ylim(thetamin, thetamax)
        
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]    
    handles2, labels2 = theta_ax.get_legend_handles_labels()
    handles2 = [h[0] for h in handles2]
    
    ax.legend(handles+handles2, labels+labels2, loc=0)
    
    ax.grid()

    plt.tight_layout();
    if(save == True):
        plt.savefig(savename)
    plt.show()    

def compute_freshwater_input(h_w, abssalmean, rhomean):
    if(len(absalmean[np.isnan(abssalmean)]) > (len(abssalmean) - 2) ):
        return 0
    else:
        min_ind = np.nanargmin(abssalmean)
        max_ind = np.nanargmax(abssalmean)
        freshwater_h[i] = h_w * ( (rhomean[s:e][max_ind] * abssalmean[s:e][max_ind]) / (rhomean[s:e][min_ind] * abssalmean[s:e][min_ind])  - 1 ) * 1e3
        print(h_w, rhomean[s:e][max_ind], abssalmean[s:e][max_ind], rhomean[s:e][min_ind], abssalmean[s:e][min_ind])
        

def get_surface_theta_sal_averages_vs_year(df, mask, years=[], thetamin=0, thetamax=0, salmin=0, salmax=0, save=False, savename="untitled.png", markersize=3, wd=10, ht=4, nmin=3, h_w=0.0, clim=False):
    if not years:
        years = np.sort(df.loc[mask, 'JULD'].dt.year.unique())
    if(clim == False):
        iter_range = len(years)
    else:
        iter_range = 1
        
    salmean = np.zeros(iter_range*12)
    abssalmean = np.zeros(iter_range*12)
    rhomean = np.zeros(iter_range*12)
    sal_sd = np.zeros(iter_range*12)
    sal_count = np.zeros(iter_range*12)
    thetamean = np.zeros(iter_range*12)
    theta_sd = np.zeros(iter_range*12)
    theta_count = np.zeros(iter_range*12)

    freshwater_h = np.zeros(iter_range)
    for i in range(iter_range):
        if(clim == False):
            yearmask = df['JULD'].dt.year == years[i]
        else:
            yearmask = df['JULD'].dt.year.isin(years)
        for j in range(12):
            monthmask = df['JULD'].dt.month == j+1
            salmean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'PSAL_ADJUSTED'].mean()
            abssalmean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'SA'].mean()
            rhomean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'DENSITY_INSITU'].mean()
            thetamean[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'CTEMP'].mean()

            sal_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'PSAL_ADJUSTED'].std()
            theta_sd[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'CTEMP'].std()
            
            sal_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'PSAL_ADJUSTED'].count()
            theta_count[i*12+j] = df.loc[df.loc[mask & yearmask & monthmask].groupby('PROFILE_NUMBER').head(1).index, 'CTEMP'].count()
            
        s,e = i*12, i*12+12
        if(np.isnan(abssalmean[s:e]).all()):
            continue
        else:
            min_ind = np.nanargmin(abssalmean[s : e])
            max_ind = np.nanargmax(abssalmean[s : e])
            if(h_w == 0.0):
                h_w = 2.0*abs(df[mask & yearmask].groupby('PROFILE_NUMBER').head(1).DEPTH.mean())
            freshwater_h[i] = h_w * ( (rhomean[s:e][max_ind] * abssalmean[s:e][max_ind]) / (rhomean[s:e][min_ind] * abssalmean[s:e][min_ind])  - 1 ) * 1e3
            print(h_w, rhomean[s:e][max_ind], abssalmean[s:e][max_ind], rhomean[s:e][min_ind], abssalmean[s:e][min_ind])
        
    salmean = ma.masked_array(salmean, mask = return_ma_nmin(sal_count, nmin) )
    sal_sd = ma.masked_array(sal_sd, mask = return_ma_nmin(sal_count, nmin) )
    thetamean = ma.masked_array(thetamean, mask = return_ma_nmin(theta_count, nmin))
    theta_sd = ma.masked_array(theta_sd, mask = return_ma_nmin(theta_count, nmin))
    
    #sal_yerror = 1.96 * sal_sd / np.sqrt(sal_count)
    sal_yerror = sal_sd
    #theta_yerror = 1.96 * theta_sd / np.sqrt(theta_count)
    theta_yerror = theta_sd
    
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] * iter_range
    timeaxis_yearticklabel = []
    if(clim == False):
        for i in range(iter_range):
            timeaxis_yearticklabel.insert(len(timeaxis_yearticklabel), str(years[i])+"\n"+str(int(freshwater_h[i]))+"mm yr$^{-1}$")
        
    timeaxis = np.arange(1, iter_range*12+1, 1)
    fig, ax = plt.subplots(figsize=(wd, ht))

    theta_ax = ax.twinx()

    if(clim == False):
        fig.subplots_adjust(bottom=0.20)
        year_ax = ax.twiny()
        year_ax.set_frame_on(True)
        year_ax.patch.set_visible(False)
        year_ax.xaxis.set_ticks_position('bottom')
        year_ax.xaxis.set_label_position('bottom')
        year_ax.spines['bottom'].set_position(('outward', 30))
        year_ax.set_xticks(np.arange(1,iter_range, 12))
        year_ax.set_xticklabels(timeaxis_yearticklabel, rotation='0')
        year_ax.set_xlim(0, timeaxis[-1]+1)
        

    theta_ax.errorbar(timeaxis, thetamean, yerr=theta_yerror, fmt='x', markersize=markersize, capsize=3, color='r', label="Pot. temp.")
    theta_ax.set_xlim(0, timeaxis[-1]+1)
    theta_ax.set_xticks(timeaxis)
    theta_ax.set_xticklabels(timeaxis_ticklabel)
    theta_ax.set_ylabel("Pot. temp. ($\\theta^o$C)")
    
    ax.errorbar(timeaxis, salmean, yerr=sal_yerror, fmt='o', markersize=markersize, capsize=3, label="salinity", color='b')
    ax.set_xlim(0, timeaxis[-1]+1)
    ax.set_xticks(timeaxis)
    ax.set_xticklabels(timeaxis_ticklabel)
    ax.set_ylabel('Salinity (Cond.)')
    
    if(salmin != 0 and salmax != 0):
        ax.set_ylim(salmin, salmax)
    if(thetamin != 0 or thetamax != 0):
        theta_ax.set_ylim(thetamin, thetamax)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    handles2, labels2 = theta_ax.get_legend_handles_labels()
    handles2 = [h[0] for h in handles2]
    ax.legend(handles+handles2, labels+labels2, loc=0)
            
    ax.grid()

    plt.tight_layout();
    if(save == True):
        plt.savefig(savename)
    plt.show()    
