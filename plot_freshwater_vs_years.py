from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib

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
                            label='$M_{glacial}$')
        
        #ax.errorbar(timeaxis, gross_fh, yerr=gross_fh_err, capsize=4, fmt='none', color='b')
        #ax.errorbar(timeaxis, fh[1][0], yerr=fh[1][1], fmt='none', color='m')
        #ax.errorbar(timeaxis, -fh[2][0], yerr=fh[2][1], fmt='none', color='g')
        ax.errorbar(timeaxis, net_glacier_fh, yerr=net_glacier_fh_err, fmt='none', color='r', capsize=4)
        #ax2.errorbar(timeaxis, net_glacier_mass, yerr=net_glacier_mass_err, fmt='none', color='k')
        ax.set_xticks(timeaxis)
        ax.set_xticklabels(xticklabel)
        ax.set_ylabel("mm yr$^{-1}$")
        ax2.set_ylabel("$M_{glacial}$ (Gta$^{-1}$)")
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


def print_climatological_fh(fh, area, verbose=False, plot=False, wd=5, ht=3, save=False, savename='untitled.png', bwd=.2, show_legend=False, ymin=0, ymax=2500, fontsize=14):
    matplotlib.rcParams.update({'font.size': fontsize})        
    gross_fh = 0.0
    gross_fh_err = 0.0
    net_glacier_fh = 0.0
    net_glacier_fh_err = 0.0
    net_glacier_mass = 0.0
    net_glacier_mass_err = 0.0
    xticklabel = []
    
    gross_fh = np.nansum(fh[0][0][0,::-1][0:6])
    gross_fh_err = np.nansum(fh[0][1][0,::-1][0:6])
    net_glacier_fh = np.nansum(fh[0][0][0,::-1][0:6]) + fh[1][0] - fh[2][0]
    net_glacier_fh_err = np.nansum(fh[0][1][0,::-1][0:6]) + fh[1][1] + fh[2][1]
    rho = 1e3
    net_glacier_mass = (net_glacier_fh * 1e-3) * (area * 1e6) * rho * 1e-12
    net_glacier_mass_err = (net_glacier_fh_err * 1e-3) * (area * 1e6) * rho * 1e-12
        
    #xticklabel.append(str(years[i])+"\n"+str(int(area[i]))+" km$^2$")

    print(fh[0][0][0,::-1][0:6].sum(), fh[1][0], fh[2][0])
    
    if(plot == True):
        fig, ax = plt.subplots(figsize=(wd, ht))
        ax2 = ax.twinx()

        lns1 = ax.bar(1, gross_fh, bwd, label='Gross $f_h$', yerr=gross_fh_err)
        lns2 = ax.bar(2, -fh[1][0], bwd, label='P-E $f_{h(P-E)}$', yerr=fh[1][1])
        lns3 = ax.bar(3, fh[2][0], bwd, label='Sea-ice $f_{h(sea ice)}$', yerr=fh[2][1])
        lns4 = ax.bar(4, net_glacier_fh, bwd, 
                           label='Glacial Runoff $f_{h(glacial)}$', yerr=net_glacier_fh_err )
        lns5 = ax2.bar(5, net_glacier_mass, bwd, yerr=net_glacier_mass_err, 
                            label='$M_{glacial}$', color="0.75")
        xticklabel = ['Gross $f_h$', '$f_{h(E-P)}$', '$f_{h(sea ice)}$', '$f_{h(glacial)}$', '$M_{glacial}$' ]
        ax.set_xticks(np.arange(1,6))
        ax.set_xticklabels(xticklabel, rotation=90)
        ax.set_ylabel("$f_h$ mm yr$^{-1}$")
        ax2.set_ylabel("$M_{glacial}$ (Gta$^{-1}$)")
        ax.set_ylim(ymin, ymax)
        ax.grid()
        yaxlim = ax.get_ylim()
        ratio1 = yaxlim[0] / (yaxlim[1] - yaxlim[0])
        y2axlim = ax2.get_ylim()
        bot2 = ratio1 * y2axlim[1] / (1 + ratio1)
        ax2.set_ylim(bot2, y2axlim[1])

        if(show_legend == True):
            #lns = lns1+lns2+lns3+lns4+lns5
            lns = lns1+lns2+lns3+lns4+lns5
            labs = [l.get_label() for l in lns]
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            lgd = ax.legend(handles+handles2, labels+labels2, loc=0, bbox_to_anchor=(.93, -0.3), fancybox=True, ncol=3)
            #lgd = ax.legend(lns, labs, loc=0, bbox_to_anchor=(1., -0.5), fancybox=True, ncol=3)
        
        plt.tight_layout()
        if(save == True):
            if(show_legend == True):
                plt.savefig(savename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                plt.savefig(savename, dpi=150)
        plt.show()
    print(net_glacier_fh_err)
    return net_glacier_fh

def plot_fh_vs_years(fh, years, wd=4, ht=2, save=False, savename="untitled.png"):
    fig, ax = plt.subplots(figsize(wd, ht) )
    ax.plot(np.arange(years[0], years[-1]+1, 1), fh, 'o')
    if(save == True):
        plt.savefig(savename, dpi=150)
