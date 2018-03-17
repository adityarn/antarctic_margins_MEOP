import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_theta_s(df, mask, wd=7, ht=4, salmin=30, salmax=36, thetamin=-3, thetamax=6, alpha=0.2, save=False, savename='untitled.png'):
    plt.figure(figsize=(wd, ht))
    thetas = df.loc[mask, 'CTEMP']
    sals = df.loc[mask, 'PSAL_ADJUSTED']
    depth = df.loc[mask, 'DEPTH']
    plt.scatter(sals, thetas, s=0.1, c=depth, alpha=alpha, vmin=-650, vmax=0)
    color_bar = plt.colorbar()
    color_bar.set_alpha(1)
    color_bar.draw_all()    
    plt.xlim(salmin, salmax)
    plt.ylim(thetamin, thetamax)
    if(save == True):
        plt.savefig(savename)
    plt.show()    

def plot_theta_s_yearly(df, mask, years=[], salmin=32.5, salmax=35.5, tempmin=-3, tempmax=1.0, alpha=1, wd=7, ht=7, s=1, save=False, savename="untitled.png"):
    fig = plt.figure(figsize=(wd, ht))
    if not years:
        years = df.loc[mask, 'JULD'].dt.year.unique()
    color = cm.Set1(np.arange(len(years)))
    for i in range(len(years)):
        year_mask = df.loc[:, 'JULD'].dt.year == years[i]
        thetas = df.loc[mask & year_mask, 'CTEMP']
        sals = df.loc[mask & year_mask, 'PSAL_ADJUSTED']
        
        plt.scatter(sals, thetas, c=color[i], s=s, label=str(years[i]), alpha=alpha)
        plt.xlim(salmin, salmax)
        plt.ylim(tempmin, tempmax)

    leg = plt.legend(loc=0, scatterpoints=1, fontsize=10)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [20]
        lh._sizes = [20]

    if(save == True):
        plt.savefig(savename)
    plt.show()    
    

    
