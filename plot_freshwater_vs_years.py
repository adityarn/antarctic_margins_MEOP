from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma

# prints the yearly freshwater flux equivalent height as computed from salinity conservation 
# in the top 100m of the water column
def print_yearly_fh(fh, years, area, verbose=False, plot=False, wd=5, ht=3, save=False, savename='untitled.png'):
    gross_fh = np.zeros(len(years))
    gross_fh_err = np.zeros(len(years))
    net_glacier_fh = np.zeros(len(years))
    net_glacier_fh_err = np.zeros(len(years))
    net_glacier_mass = np.zeros(len(years))
    net_glacier_mass_err = np.zeros(len(years))
    xticklabel = []
    
    for i in range(len(years)):
        gross_fh[i] = np.nansum(fh[0][0][i, ::-1][0:6])
        gross_fh_err[i] = np.nansum(fh[0][1][i, ::-1][0:6])
        net_glacier_fh[i] = np.nansum(fh[0][0][i, ::-1][0:6]) + fh[1][0][i] - fh[2][0][i]
        net_glacier_fh_err[i] = np.nansum(fh[0][1][i, ::-1][0:6]) + fh[1][1][i] + fh[2][1][i]
        rho = 1e3
        net_glacier_mass[i] = (net_glacier_fh[i] * 1e-3) * (area[i] * 1e6) * rho * 1e-12
        net_glacier_mass_err[i] = (net_glacier_fh_err[i] * 1e-3) * (area[i] * 1e6) * rho * 1e-12
        
        xticklabel.append(str(years[i])+"\n"+str(int(area[i]))+" km$^2$")
        if(verbose==True):
            print(years[i], fh[0][0][i, ::-1][0:6].sum(), fh[1][0][i], fh[2][0][i])
    if(plot == True):
        fig, ax = plt.subplots(figsize=(wd, ht))
        ax2 = ax.twinx()
        timeaxis = np.arange(len(years))
        lns1 = ax.plot(timeaxis, gross_fh, '.', label='Gross $f_h$', color='b')
        #lns2 = ax.plot(timeaxis, fh[1][0], '.', label='E-P $f_{h(E-P)}$', color='m')
        #lns3 = ax.plot(timeaxis, -fh[2][0], '.', label='Sea-ice $f_{h(sea ice)}$', color='g')
        lns4 = ax.plot(timeaxis, net_glacier_fh, '.', 
                           label='Glacial Runoff $f_{h(glacial)}$', color='r')
        lns5 = ax2.plot(timeaxis, net_glacier_mass, 'x', color='k', 
                            label='Glacial Runoff Mass')
        
        #ax.errorbar(timeaxis, gross_fh, yerr=gross_fh_err, capsize=4, fmt='none', color='b')
        #ax.errorbar(timeaxis, fh[1][0], yerr=fh[1][1], fmt='none', color='m')
        #ax.errorbar(timeaxis, -fh[2][0], yerr=fh[2][1], fmt='none', color='g')
        ax.errorbar(timeaxis, net_glacier_fh, yerr=net_glacier_fh_err, fmt='none', color='r', capsize=4)
        #ax2.errorbar(timeaxis, net_glacier_mass, yerr=net_glacier_mass_err, fmt='none', color='k')
        ax.set_xticks(timeaxis)
        ax.set_xticklabels(xticklabel)
        ax.set_ylabel("mm yr$^{-1}$")
        ax2.set_ylabel("Gta$^{-1}$")
        ax.grid()
        
        #lns = lns1+lns2+lns3+lns4+lns5
        lns = lns1+lns4+lns5
        labs = [l.get_label() for l in lns]
        lgd = ax.legend(lns, labs, loc=0, bbox_to_anchor=(1., -0.2), fancybox=True, ncol=3)
        
        plt.tight_layout()
        if(save == True):
            plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
    print(net_glacier_fh_err)
    return net_glacier_fh

def plot_fh_vs_years(fh, years, wd=4, ht=2, save=False, savename="untitled.png"):
    fig, ax = plt.subplots(figsize(wd, ht) )
    ax.plot(np.arange(years[0], years[-1]+1, 1), fh, 'o')
    if(save == True):
        plt.savefig(savename, dpi=150)
