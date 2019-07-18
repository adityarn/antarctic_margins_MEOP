from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib
from matplotlib import gridspec

def compute_monthly_conditional_prob_weighted(dfwm, wm1='DSW', wm2='CDW', save=False, savename='untitled.png', 
                                     show=True, width=90/25.4, height=90/25.4):
    matplotlib.rcParams.update({'font.size': 8})
    prob_DSW = np.zeros(12)
    prob_noDSW = np.zeros(12)
    prob_CDW = np.zeros(12)
    prob_noCDW = np.zeros(12)
    
    prob_DSW_cap_CDW = np.zeros(12)
    prob_DSW_cap_noCDW = np.zeros(12)
    prob_CDW_cap_noDSW = np.zeros(12)
    
    prob_DSW_cond_CDW = np.zeros(12)
    prob_DSW_cond_noCDW = np.zeros(12)
    prob_CDW_cond_DSW = np.zeros(12)
    prob_CDW_cond_noDSW = np.zeros(12)
    
    wm1gt0 = dfwm[wm1] > 0
    wm2gt0 = dfwm[wm2] > 0
    DSW_cap_CDW = wm1gt0 & wm2gt0
    DSW_cap_noCDW = wm1gt0 & ~wm2gt0
    CDW_cap_noDSW = ~wm1gt0 & wm2gt0
    
    noNull = ~dfwm["DSW"].isnull()
    
    for i in range(12):
        monthMask = (dfwm.month.isin([i+1]) & noNull)
    
        prob_DSW[i] = np.nansum(dfwm.loc[monthMask, wm1]) / np.abs(np.nansum(dfwm.loc[monthMask, 'zlowest']))
        prob_CDW[i] = np.nansum(dfwm.loc[monthMask, wm2]) / np.abs(np.nansum(dfwm.loc[monthMask, 'zlowest']))
        prob_noCDW[i] = 1 - prob_CDW[i]
        prob_noDSW[i] = 1 - prob_DSW[i]
        
        prob_DSW_cap_CDW[i] = (np.nansum(dfwm.loc[monthMask & DSW_cap_CDW, wm1]) + np.nansum(dfwm.loc[monthMask & DSW_cap_CDW, wm2]) ) / np.abs(np.nansum(dfwm.loc[monthMask, 'zlowest']))
        prob_DSW_cap_noCDW[i] = np.nansum(dfwm.loc[monthMask & DSW_cap_noCDW, wm1]) / np.abs(np.nansum(dfwm.loc[monthMask, 'zlowest']))
        prob_CDW_cap_noDSW[i] = np.nansum(dfwm.loc[monthMask & CDW_cap_noDSW, wm2]) / np.abs(np.nansum(dfwm.loc[monthMask, 'zlowest']))
        
    prob_DSW_cond_CDW = prob_DSW_cap_CDW / prob_CDW
    prob_DSW_cond_noCDW = prob_DSW_cap_noCDW / prob_noCDW
    
    prob_CDW_cond_DSW = prob_DSW_cap_CDW / prob_DSW
    prob_CDW_cond_noDSW = prob_CDW_cap_noDSW / prob_noDSW
    
    timeaxis = np.arange(1,13,1)
    wd = 0.2
    wm1 = wm1
    wm2 = wm2
    
    fig = plt.figure(1, figsize=(width, height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.2])
    ax = plt.subplot(gs[0,0])
    
    ax.bar(timeaxis-1.5*wd, prob_DSW_cond_CDW, wd, label='P('+wm1+' | '+wm2+')')
    ax.bar(timeaxis-wd*0.5, prob_DSW_cond_noCDW, wd, label='P('+wm1+' | no '+wm2+')')
    
    ax.bar(timeaxis+wd*0.5, prob_CDW_cond_DSW, wd, label='P('+wm2+' | '+wm1+')')
    plt.bar(timeaxis+wd*1.5, prob_CDW_cond_noDSW, wd, label='P('+wm2+' | no '+wm1+')')
    
    ax.set_xticks(timeaxis)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticklabels(timeaxis_ticklabel)
    
    handles, labels = ax.get_legend_handles_labels()
    axlegend = plt.subplot(gs[1,0])
    axlegend.legend(handles, labels, ncol=2, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    ax.grid()
    
    if save:
        plt.savefig(savename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    print(' P('+wm1+' | '+wm2+')=',prob_DSW_cond_CDW.mean(),"\n", 'P('+wm1+' | no '+wm2+')=', prob_DSW_cond_noCDW.mean(), 
          "\n", 'P('+wm2+' | '+wm1+')=', prob_CDW_cond_DSW.mean(), "\n", 
          'P('+wm2+' | no '+wm1+')=',prob_CDW_cond_noDSW.mean())



def compute_prob_unweighted(dfwm, wm1='DSW', wm2='CDW', save=False, savename='untitled.png', 
                                     show=True, width=90/25.4, height=90/25.4):

    wm1gt0 = dfwm[wm1] > 0
    dfwm.loc[wm1gt0, wm1+"_flag"] = 1
    dfwm.loc[~wm1gt0, wm1+"_flag"] = 0

    wm2gt0 = dfwm[wm2] > 0
    dfwm.loc[wm2gt0, wm2+"_flag"] = 1
    dfwm.loc[~wm2gt0, wm2+"_flag"] = 0

    matplotlib.rcParams.update({'font.size': 8})
    prob_DSW = np.zeros(12)
    prob_noDSW = np.zeros(12)
    prob_CDW = np.zeros(12)
    prob_noCDW = np.zeros(12)
    
    prob_DSW_cap_CDW = np.zeros(12)
    prob_DSW_cap_noCDW = np.zeros(12)
    prob_CDW_cap_noDSW = np.zeros(12)
    
    prob_DSW_cond_CDW = np.zeros(12)
    prob_DSW_cond_noCDW = np.zeros(12)
    prob_CDW_cond_DSW = np.zeros(12)
    prob_CDW_cond_noDSW = np.zeros(12)
    
    DSW_cap_CDW = wm1gt0 & wm2gt0
    DSW_cap_noCDW = wm1gt0 & ~wm2gt0
    CDW_cap_noDSW = ~wm1gt0 & wm2gt0
    
    noNull = ~dfwm["DSW"].isnull()
    
    for i in range(12):
        monthMask = (dfwm.month.isin([i+1]) & noNull)
        totalLength = float(len(dfwm.loc[monthMask]))
        
        prob_DSW[i] = np.nansum(dfwm.loc[monthMask, wm1+"_flag"]) / totalLength
        prob_CDW[i] = np.nansum(dfwm.loc[monthMask, wm2+"_flag"]) / totalLength
        prob_noCDW[i] = 1 - prob_CDW[i]
        prob_noDSW[i] = 1 - prob_DSW[i]
        
        prob_DSW_cap_CDW[i] = (np.nansum(dfwm.loc[monthMask & DSW_cap_CDW, wm1+"_flag"]) ) / totalLength
        prob_DSW_cap_noCDW[i] = np.nansum(dfwm.loc[monthMask & DSW_cap_noCDW, wm1+"_flag"]) / totalLength
        prob_CDW_cap_noDSW[i] = np.nansum(dfwm.loc[monthMask & CDW_cap_noDSW, wm2+"_flag"]) / totalLength
        
    prob_DSW_cond_CDW = prob_DSW_cap_CDW / prob_CDW
    prob_DSW_cond_noCDW = prob_DSW_cap_noCDW / prob_noCDW
    
    prob_CDW_cond_DSW = prob_DSW_cap_CDW / prob_DSW
    prob_CDW_cond_noDSW = prob_CDW_cap_noDSW / prob_noDSW
    
    timeaxis = np.arange(1,13,1)
    wd = 0.2
    wm1 = wm1
    wm2 = wm2
    
    fig = plt.figure(1, figsize=(width, height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.2])
    ax = plt.subplot(gs[0,0])
    
    ax.bar(timeaxis-1.5*wd, prob_DSW_cond_CDW, wd, label='P('+wm1+' | '+wm2+')')
    ax.bar(timeaxis-wd*0.5, prob_DSW_cond_noCDW, wd, label='P('+wm1+' | no '+wm2+')')
    
    ax.bar(timeaxis+wd*0.5, prob_CDW_cond_DSW, wd, label='P('+wm2+' | '+wm1+')')
    plt.bar(timeaxis+wd*1.5, prob_CDW_cond_noDSW, wd, label='P('+wm2+' | no '+wm1+')')
    
    ax.set_xticks(timeaxis)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticklabels(timeaxis_ticklabel)
    
    handles, labels = ax.get_legend_handles_labels()
    axlegend = plt.subplot(gs[1,0])
    axlegend.legend(handles, labels, ncol=2, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    ax.grid()
    
    if save:
        plt.savefig(savename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    print(' P('+wm1+' | '+wm2+')=',prob_DSW_cond_CDW.mean(),"\n", 'P('+wm1+' | no '+wm2+')=', prob_DSW_cond_noCDW.mean(), 
          "\n", 'P('+wm2+' | '+wm1+')=', prob_CDW_cond_DSW.mean(), "\n", 
          'P('+wm2+' | no '+wm1+')=',prob_CDW_cond_noDSW.mean())


def compute_prob_unweighted_bootstrapped(dfwm, save=False, savename='untitled.png', reps=10000,
                                        show=True, width=90/25.4, height=90/25.4):
    matplotlib.rcParams.update({'font.size': 8})        # setting fontsize for plot elements        
    wm1='DSW'
    wm2='CDW'
    prob_DSW_cond_CDW = np.zeros(2)
    prob_DSW_cond_CDW_yerr = np.zeros((2,2))
    prob_DSW_cond_CDW_CI = np.zeros((2,2))
    
    prob_DSW_cond_noCDW = np.zeros(2)
    prob_DSW_cond_noCDW_yerr = np.zeros((2,2))
    prob_DSW_cond_noCDW_CI = np.zeros((2,2))

    prob_CDW_cond_DSW = np.zeros(2)
    prob_CDW_cond_DSW_yerr = np.zeros((2,2))
    prob_CDW_cond_DSW_CI = np.zeros((2,2))

    prob_CDW_cond_noDSW = np.zeros(2)
    prob_CDW_cond_noDSW_yerr = np.zeros((2,2))
    prob_CDW_cond_noDSW_CI = np.zeros((2,2))

    season = [[12,1,2,3,4,5], [6,7,8,9,10,11]]
    for i in range(2):
        selMask = ( (dfwm.month.isin(season[i]) ) & (dfwm["DSW"].notnull()) )
        totalLength = float(len(dfwm.loc[selMask]))
        seldDSW = dfwm[selMask].DSW.values
        seldCDW = dfwm[selMask].CDW.values
        
        prob_DSW_cap_NoCDW = len(seldDSW[seldCDW == 0].nonzero()[0]) / totalLength # P(DSW intersection noCDW)
        prob_noCDW = len(seldCDW[seldCDW == 0]) / totalLength
        prob_DSW_cond_noCDW[i] = prob_DSW_cap_NoCDW / prob_noCDW

        randidxr = np.random.choice(len(seldDSW[seldCDW == 0]), (len(seldDSW[seldCDW==0]), reps), replace=True)
        MC_DSW_cap_NoCDW = seldDSW[seldCDW==0][randidxr]
        MC_prob_DSW_cap_noCDW = np.count_nonzero(MC_DSW_cap_NoCDW, axis=0) / totalLength

        randidxr = np.random.choice(len(seldCDW), (len(seldCDW), reps), replace=True)
        MC_CDW = seldCDW[randidxr]
        MC_prob_noCDW = ma.count(ma.masked_array(MC_CDW, mask=~(MC_CDW==0) ), axis=0) / totalLength
        MC_prob_DSW_cond_noCDW = MC_prob_DSW_cap_noCDW / MC_prob_noCDW
        prob_DSW_cond_noCDW_yerr[i] = abs(prob_DSW_cond_noCDW[i] - np.percentile(MC_prob_DSW_cond_noCDW, [2.5, 97.5]))
        prob_DSW_cond_noCDW_CI[i] = np.percentile(MC_prob_DSW_cond_noCDW, [2.5, 97.5])

        prob_CDW_cap_NoDSW = len(seldCDW[seldDSW == 0].nonzero()[0]) / float(totalLength)
        prob_noDSW = len(seldDSW[seldDSW == 0]) / float(totalLength)
        prob_CDW_cond_noDSW[i] = prob_CDW_cap_NoDSW / prob_noDSW

        randidxr = np.random.choice(len(seldCDW[seldDSW == 0]), (len(seldCDW[seldDSW==0]), reps), replace=True)
        MC_CDW_cap_NoDSW = seldCDW[seldDSW==0][randidxr]
        MC_prob_CDW_cap_noDSW = np.count_nonzero(MC_CDW_cap_NoDSW, axis=0) / totalLength

        randidxr = np.random.choice(len(seldDSW), (len(seldDSW), reps), replace=True)
        MC_DSW = seldDSW[randidxr]
        MC_prob_noDSW = ma.count(ma.masked_array(MC_DSW, mask=~(MC_DSW==0) ), axis=0) / totalLength
        MC_prob_CDW_cond_noDSW = MC_prob_CDW_cap_noDSW / MC_prob_noDSW
        prob_CDW_cond_noDSW_yerr[i] = abs(prob_CDW_cond_noDSW[i] - np.percentile(MC_prob_CDW_cond_noDSW, [2.5, 97.5]))
        prob_CDW_cond_noDSW_CI[i] = np.percentile(MC_prob_CDW_cond_noDSW, [2.5, 97.5])

        prob_DSW_cap_CDW = len(seldDSW[seldCDW > 0].nonzero()[0]) / totalLength
        prob_CDW = len(seldCDW.nonzero()[0]) / totalLength
        prob_DSW_cond_CDW[i] = prob_DSW_cap_CDW / prob_CDW

        randidxr = np.random.choice(len(seldDSW[seldCDW > 0]), (len(seldDSW[seldCDW > 0]), reps), replace=True)
        MC_DSW_cap_CDW = seldDSW[seldCDW > 0][randidxr]
        MC_prob_DSW_cap_CDW = np.count_nonzero(MC_DSW_cap_CDW, axis=0) / totalLength

        MC_prob_CDW = ma.count(ma.masked_array(MC_CDW, mask=(MC_CDW==0) ), axis=0) / totalLength
        MC_prob_DSW_cond_CDW = MC_prob_DSW_cap_CDW / MC_prob_CDW
        prob_DSW_cond_CDW_yerr[i] = abs(prob_DSW_cond_CDW[i] - np.percentile(MC_prob_DSW_cond_CDW, [2.5, 97.5]))
        prob_DSW_cond_CDW_CI[i] = np.percentile(MC_prob_DSW_cond_CDW, [2.5, 97.5])

        prob_CDW_cap_DSW = len(seldCDW[seldDSW > 0].nonzero()[0]) / totalLength
        prob_DSW = len(seldDSW[seldDSW > 0]) / totalLength
        prob_CDW_cond_DSW[i] = prob_CDW_cap_DSW / prob_DSW

        randidxr = np.random.choice(len(seldCDW[seldDSW > 0]), (len(seldCDW[seldDSW > 0]), reps), replace=True)
        MC_CDW_cap_DSW = seldCDW[seldDSW > 0][randidxr]
        MC_prob_CDW_cap_DSW = np.count_nonzero(MC_CDW_cap_DSW, axis=0) / totalLength

        MC_prob_DSW = ma.count(ma.masked_array(MC_DSW, mask=(MC_DSW==0) ), axis=0) / totalLength
        MC_prob_CDW_cond_DSW = MC_prob_CDW_cap_DSW / MC_prob_DSW
        prob_CDW_cond_DSW_yerr[i] = abs(prob_CDW_cond_DSW[i] - np.percentile(MC_prob_CDW_cond_DSW, [2.5, 97.5]))
        prob_CDW_cond_DSW_CI[i] = np.percentile(MC_prob_CDW_cond_DSW, [2.5, 97.5])


    timeaxis = np.array([1,2])
    fig = plt.figure(1, figsize=(width, height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.2])
    ax = plt.subplot(gs[0,0])
    wth = 0.1
    print(prob_DSW_cond_CDW, prob_DSW_cond_CDW_yerr)
    ax.errorbar(timeaxis-1.5*wth, prob_DSW_cond_CDW, yerr=prob_DSW_cond_CDW_yerr.T, label='P('+wm1+' | '+wm2+')', color="aqua", fmt="s", capsize=3)
    ax.errorbar(timeaxis-wth*0.5, prob_DSW_cond_noCDW, yerr=prob_DSW_cond_noCDW_yerr.T, label='P('+wm1+' | no '+wm2+')', color="b", fmt="o", capsize=3)
    
    ax.errorbar(timeaxis+wth*0.5, prob_CDW_cond_DSW, yerr=prob_CDW_cond_DSW_yerr.T, label='P('+wm2+' | '+wm1+')', color="darkorange", fmt="v", capsize=3)
    ax.errorbar(timeaxis+wth*1.5, prob_CDW_cond_noDSW, yerr=prob_CDW_cond_noDSW_yerr.T, label='P('+wm2+' | no '+wm1+')', color="r", fmt="^", capsize=3)
    
    ax.set_xticks(timeaxis)
    timeaxis_ticklabel = ['summer', 'winter']
    ax.set_xticklabels(timeaxis_ticklabel)
    
    handles, labels = ax.get_legend_handles_labels()
    axlegend = plt.subplot(gs[1,0])
    axlegend.legend(handles, labels, ncol=2, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    ax.set_xlim(0.3, 2.75)
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.1,0.1))

    ax.grid()
    
    if save:
        plt.savefig(savename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    print(' P('+wm1+' | '+wm2+')=',prob_DSW_cond_CDW,prob_DSW_cond_CDW_CI,
          "\n", 'P('+wm1+' | no '+wm2+')=', prob_DSW_cond_noCDW, prob_DSW_cond_noCDW_CI, 
          "\n", 'P('+wm2+' | '+wm1+')=', prob_CDW_cond_DSW, prob_CDW_cond_DSW_CI,
          "\n", 'P('+wm2+' | no '+wm1+')=',prob_CDW_cond_noDSW, prob_CDW_cond_noDSW_CI)
    return prob_DSW_cond_noCDW, prob_DSW_cond_noCDW_CI, prob_DSW_cond_noCDW_yerr

