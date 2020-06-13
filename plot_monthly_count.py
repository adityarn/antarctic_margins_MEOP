import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors
import matplotlib

def time_vs_count(df, show=False, save=False, savename="untitled.png", wd=8, ht=3):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(wd, ht))
    months = np.arange(1,13,1)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    count = np.zeros(12)

    for i in range(12):
        month_mask = df['JULD'].dt.month.isin([months[i]])
        count[i] = len(df.loc[month_mask, 'PROFILE_NUMBER'].unique())

    ax.plot(months, count, 'o')
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.grid()
    ax.set_ylabel("No of Profiles")
    ax.set_xlabel("Months")
    if(show == True):
        plt.show();
    plt.close();


def plot_counts_profs(ax, df, WSO_source, WSO_prod, boxes, title, fontsize=8):
    
    matplotlib.rcParams.update({'font.size': fontsize})    
    monthnos = np.arange(1,13,1)
    months = ['J','F','M','A','M','J','J','A','S','O','N','D']

    #fig, axarr = plt.subplots(row, col, figsize=(wd,ht))

    nprof_all = np.zeros(12)
    nprof_source = np.zeros(12)
    nprof_prod = np.zeros(12)

    for i in range(12):
        nprof_all[i] = len(df.loc[(df['JULD'].dt.month.isin([monthnos[i]]) & boxes),'PROFILE_NUMBER'].unique())
        nprof_source[i] = len(df.loc[(df['JULD'].dt.month.isin([monthnos[i]]) &WSO_source & boxes), 'PROFILE_NUMBER'].unique())
        nprof_prod[i] = len(df.loc[(df['JULD'].dt.month.isin([monthnos[i]]) &WSO_prod & boxes), 'PROFILE_NUMBER'].unique())

    ax_sp = ax.twinx()
    wd=0.2
    rects_all = ax.bar(np.arange(12), nprof_all, wd, color='0.25', label='all')
    rects_source = ax_sp.bar(np.arange(12)+wd, nprof_source/nprof_all*100, wd, color='r', label='source')
    rects_prod = ax_sp.bar(np.arange(12)+wd*2, nprof_prod/nprof_all*100, wd, color='b', label='prod')
    ax.set_xticks(np.arange(12) + wd*.5)
    ax.set_xticklabels(months)
    ax.set_ylim(1e-1, 1e4)
    ax_sp.set_ylim(0,100)
    #if(n == 0):
        #ax.set_ylabel("count of all profiles")
    #if(n == col-1):
        #ax_sp.set_ylabel("% source/product")
    #ax.set_title(titles[j])
    ax.set_yscale("log")
    ## handles, labels = ax.get_legend_handles_labels()
    ## handles2, labels2 = ax_sp.get_legend_handles_labels()
        
        #if(j == 0):
            #ax.legend(handles+handles2, labels+labels2, loc=0)
    ax.set_title(title)
    
    ## if(save==True):
    ##     plt.savefig(savename, dpi=150)
    return ax_sp


def plot_yearly_counts_profs(df, WSO_source, WSO_prod, boxes=[], 
                            titles=['box2', 'box4', 'box5', 'box7', 'box9', 'box10'],
                            row=2, col=3, wd=12, ht=10, save=False, savename="Untitled.png" ):
    
    years = np.arange(2004,2016,1)
    

    fig, axarr = plt.subplots(row, col, figsize=(wd,ht))


    for j in range(len(boxes)):
        nprof_all = np.zeros(len(years))
        nprof_source = np.zeros(len(years))
        nprof_prod = np.zeros(len(years))

        for i in range(len(years)):
            nprof_all[i] = len(df.loc[(df['JULD'].dt.year.isin([years[i]]) & boxes[j]),'PROFILE_NUMBER'].unique())
            nprof_source[i] = len(df.loc[(df['JULD'].dt.year.isin([years[i]]) &WSO_source & boxes[j]), 'PROFILE_NUMBER'].unique())
            nprof_prod[i] = len(df.loc[(df['JULD'].dt.year.isin([years[i]]) &WSO_prod & boxes[j]), 'PROFILE_NUMBER'].unique())

        m = int(j/col)
        n = int(j%col)
        print(m,n)
        ax = axarr[m,n]
        ax_sp = ax.twinx()
        wd=0.2
        rects_all = ax.bar(np.arange(12), nprof_all, wd, color='0.25', label='all')
        rects_source = ax_sp.bar(np.arange(12)+wd, nprof_source/nprof_all*100, wd, color='r', label='source')
        rects_prod = ax_sp.bar(np.arange(12)+wd*2, nprof_prod/nprof_all*100, wd, color='b', label='prod')
        ax.set_xticks(np.arange(12) + wd*.5)
        ax.set_xticklabels(years, rotation='90')
        ax.set_ylim(1e-1, 2e4)
        ax_sp.set_ylim(0,100)
        if(n == 0):
            ax.set_ylabel("count of all profiles")
        if(n == col-1):
            ax_sp.set_ylabel("% occurence of source or product \n in total count of all profiles")
        ax.set_title(titles[j])
        ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_sp.get_legend_handles_labels()
        
        if(j == 0):
            ax.legend(handles+handles2, labels+labels2, loc=0)
    plt.tight_layout()
    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show()
