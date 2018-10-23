import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import gsw
import matplotlib

def plot_theta_s(ax, df, mask, title="title",salmin=30, salmax=36, thetamin=-3, thetamax=6, alpha=1.0, s=15, templine=False, sig_lines=[], fontsize=8,
                 sig_line_annot= [], colorbar_show=False, scat_vmin=0, scat_vmax=650, theta_ticks=[], sal_ticks=[], show_legend=False):
    matplotlib.rcParams.update({'font.size': fontsize})    
    #fig, ax = plt.subplots(figsize=(wd, ht))
    
    thetas = df.loc[mask, 'CTEMP']
    sals = df.loc[mask, 'PSAL_ADJUSTED']
    press = df.loc[mask, 'PRES_ADJUSTED']
    
    SC = ax.scatter(sals, thetas, s=s, c=press, alpha=alpha, vmin=scat_vmin, vmax=scat_vmax)

    if not theta_ticks:
        pass
    else:
        ax.set_yticks(theta_ticks)
        
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
    
    
    SA = gsw.SA_from_SP(np.linspace(xlim[0], xlim[-1], 300), 0, df['LONGITUDE'].mean(), df['LATITUDE'].mean())
    CT = np.linspace(ylim[0], ylim[-1], 300)# - abs(ylim[-1])*0.1, 300)
    
    SS,TT = np.meshgrid(SA, CT)
    sigma00 = gsw.density.sigma0(SS, TT)

    if not sig_lines:
        sig_lines = list(np.linspace(27,27.9, 10))
        
    CS = ax.contour(SS, TT, sigma00, sig_lines, colors='k')
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
        ax.axhline(y=-1.9, linestyle="--", linewidth=1)
        ax.annotate("-1.9", xy=(xlim[0] + (xlim[-1] - xlim[0])*0.01, -1.88), fontsize=fontsize)
    ax.set_ylabel("CT ($^o$C)")
    ax.set_xlabel("Salinity (PSU)")
    ax.grid()
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
    

    
