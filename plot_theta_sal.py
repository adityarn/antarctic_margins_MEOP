import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import gsw
import matplotlib
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

sys.path.insert(0, "/usr/local/MATLAB/R2018a/extern/engines/python/build/lib.linux-x86_64-2.7")
import matlab.engine

def plot_theta_s(ax, df, mask, title="title",salmin=30, salmax=36, thetamin=-3, thetamax=6, alpha=1.0, s=15, templine=False, sig_lines=[], fontsize=8,
                 sig_line_annot= [], colorbar_show=False, scat_vmin=0, scat_vmax=650, theta_ticks_major=[], theta_ticks_minor=[], sal_ticks=[], show_legend=False):
    matplotlib.rcParams.update({'font.size': fontsize})    
    #fig, ax = plt.subplots(figsize=(wd, ht))

    sel_sal35 = (df.PSAL_ADJUSTED > 35)
    sel_tags = df.PLATFORM_NUMBER.isin(df.loc[sel_sal35, "PLATFORM_NUMBER"].unique())
    
    # sals = df[sel_tags & mask].drop_duplicates(["PSAL_ADJUSTED", "CTEMP"]).PSAL_ADJUSTED
    # thetas = df[sel_tags & mask].drop_duplicates(["PSAL_ADJUSTED", "CTEMP"]).CTEMP
    # ax.scatter(sals, thetas, s=0.1, c='0.75', alpha=0.25, vmin=scat_vmin, vmax=scat_vmax)

    thetas = df.loc[mask & ~sel_tags, 'CTEMP']
    sals = df.loc[mask & ~sel_tags, 'PSAL_ADJUSTED']
    press = df.loc[mask & ~sel_tags, 'PRES_ADJUSTED']
    SC = ax.scatter(sals, thetas, s=s, c=press, alpha=alpha, vmin=scat_vmin, vmax=scat_vmax)
    

    if theta_ticks_major:
        ax.set_yticks(theta_ticks_major)
    if theta_ticks_minor:
        ax.set_yticks(theta_ticks_minor, minor=True)
        
    if not sal_ticks:
        pass
    else:
        ax.set_xticks(sal_ticks )
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    ax.set_xlim(salmin, salmax)
    ax.set_ylim(thetamin, thetamax)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    SP = np.linspace(xlim[0], xlim[-1], 300)
    P = np.zeros(len(SP))
    SA = gsw.SA_from_SP(SP, 0, df['LONGITUDE'].mean(), df['LATITUDE'].mean())
    CT = np.linspace(ylim[0], ylim[-1], 300)# - abs(ylim[-1])*0.1, 300)
    CT_freezing = gsw.CT_freezing(SA, P, 0)
    
    SS,TT = np.meshgrid(SA, CT)
    sigma00 = gsw.sigma0(SS, TT)
    
    SPmesh, TT = np.meshgrid(SP, CT)
    eng = matlab.engine.start_matlab()
    eng.addpath('/media/work/MyPrograms/EOS80/')
    eng.addpath('/media/work/MyPrograms/EOS80/library/')
    SPmesh_mat = matlab.double(SPmesh.tolist())
    TT_mat = matlab.double(TT.tolist())
    gamman_mesh = eng.eos80_legacy_gamma_n(SPmesh_mat, TT_mat, 0.0, np.asscalar(df['LONGITUDE'].mean()), np.asscalar(df['LATITUDE'].mean()) );

    if not sig_lines:
        sig_lines = list(np.linspace(27,27.9, 10))
        
    CS = ax.contour(SPmesh, TT, gamman_mesh, sig_lines, colors='k')
    #plt.clabel(CS, colors='firebrick', fontsize=12, fmt="%3.2f")
    
    ## mask = []
    ## for i in range(len(sig_lines)):
    ##     mask.append( np.isclose(sigma00, sig_lines[i], atol=1e-4) )
    ##     if (mask[i].any() == True):
    ##         ax.plot(SS[mask[i]], TT[mask[i]], color='k', linewidth=2)
    ##         yaxislen = len(TT[mask[i]])
            
            ## if not sig_line_annot:
            ##     if(len(sig_lines) == 2):
            ##         if(i == 0):
            ##             ax.annotate(str(round(sig_lines[i], 3) ), xy=( SS[mask[i]][0] - (SS[mask[i]][0] - xlim[0])*0.1, TT[mask[i]][-int(yaxislen*0.2)]), rotation=80 )
            ##         if(i == 1):
            ##             ax.annotate(str(round(sig_lines[i], 3) ), xy=( SS[mask[i]][0] + (SS[mask[i]][0] - xlim[0])*0.1, TT[mask[i]][-int(yaxislen*0.2)]), rotation=80 )
            ##     else:
            ##         ax.annotate(str(round(sig_lines[i], 3) ), xy=(SS[mask[i]][-1], TT[mask[i]][-int(yaxislen*0.2)]), rotation=80 )
            ## else:
            ##     ax.annotate(str(round(sig_lines[i], 3) ), xy=(sig_line_annot[i][0], sig_line_annot[i][1]), rotation=80 )

    if(show_legend == True):
        colorbar_ticks = list(np.round(np.arange(scat_vmin, scat_vmax+1, 100) , 0))
        color_bar = plt.colorbar(SC, extend='max', orientation="horizontal", pad=0.3)
        color_bar.set_alpha(1)
        color_bar.set_ticks(colorbar_ticks)
        color_bar.set_ticklabels(colorbar_ticks)
        color_bar.set_label("dbar")
        color_bar.draw_all()    

    #ax.annotate(str())
    #ax.plot(SS[mask_sig_upper], TT[mask_sig_upper], color='k')
        
    if(templine == True):
        ax.plot(SP, CT_freezing, linestyle="--", linewidth=1, color="b")
        ax.axhline(y=0, linestyle="--", linewidth=1, color="r")
        #ax.axhline(y=-1.9, linestyle="--", linewidth=1)
        #ax.annotate("-1.9", xy=(xlim[0] + (xlim[-1] - xlim[0])*0.01, -1.88), fontsize=fontsize)
    ax.set_ylabel("CT ($^o$C)")
    ax.set_xlabel("Salinity (PSU)")
    ax.grid(which='both')
    ax.set_title(title)
    plt.tight_layout()
    return SC
    ## if(save == True):
    ##     plt.savefig(savename, dpi=150)
    #plt.show()    


### IGNORE FUNCTIONS BELOW, NOT USED IN CURRENT ANALYSIS ###########################################################################################################333
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
    

    
