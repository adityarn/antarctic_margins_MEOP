import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import pdb
from matplotlib import ticker, cm
from scipy import stats
from scipy.stats import pearsonr
import numpy.ma as ma


###################################################################
###################################################################
############### BOOTSTRAPPER FUNCTION #############################
def resample_depthbinData_profWise(timeDepthSliced, reps=1000):
    sliceArray = timeDepthSliced.loc[:, ["PSAL_ADJUSTED", "CTEMP", "gamman"] ].values
    try:
        randidxr = np.random.choice(len(sliceArray), (len(sliceArray), reps), replace=True )
        DSWcount = np.count_nonzero( ((sliceArray[randidxr][:,:, 2] > 28.27) & (sliceArray[randidxr][:, :, 0] > 34.5) & (sliceArray[randidxr][:,:, 1] <= -1.8) & (sliceArray[randidxr][:,:, 1] >= -1.9)), axis=0 ) / float(len(sliceArray))
        lsswCount = np.count_nonzero( ((sliceArray[randidxr][:,:, 0] >= 34.3) & (sliceArray[randidxr][:,:, 0] <= 34.4) & (sliceArray[randidxr][:,:, 1] <= -1.5) & (sliceArray[randidxr][:,:, 1] > -1.9) ), axis=0 ) / float(len(sliceArray))
        ISWcount = np.count_nonzero( (sliceArray[randidxr][:,:,1] < -1.9), axis=0) / float(len(sliceArray))
        CDWcount = np.count_nonzero( ((sliceArray[randidxr][:,:,0] >= 34.5) & (sliceArray[randidxr][:,:,1] >= 0.0)), axis=0 ) / float(len(sliceArray))
        mCDWcount = np.count_nonzero( ((sliceArray[randidxr][:,:,1] > -1.8) & (sliceArray[randidxr][:,:,1] < 0) & (sliceArray[randidxr][:,:,2] > 28) & (sliceArray[randidxr][:,:,2] < 28.27)) , axis=0) / float(len(sliceArray)) # Williams 2016
        
        return np.array([DSWcount, lsswCount, ISWcount, CDWcount, mCDWcount])
    except:
        return np.full_like(np.zeros((5,reps)), np.nan)
#order of return: DSW, lssw, ISW, mCDW, CDW
###################################################################
###################################################################


def waterMassThickness_bootstrapper_profileWise(axwmb,axdod, df, ymin=0, ymax=None, yticks=[], zbin=20, wd = 0.1, fontsize=8, showlegend=False, retValue=False, ymax_dod=None, yticks_dod=[], reps=100, markersize=3):
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements    

    DSWthickness = np.zeros(12)
    DSW_CI = np.zeros((12,2))
    DSW_yerr = np.zeros((12,2))    

    lsswthickness = np.zeros(12)
    lssw_CI = np.zeros((12,2))
    lssw_yerr = np.zeros((12,2))    

    ISWthickness = np.zeros(12)
    ISW_CI = np.zeros((12,2))
    ISW_yerr = np.zeros((12,2))    

    CDWthickness = np.zeros(12)
    CDW_CI = np.zeros((12,2))
    CDW_yerr = np.zeros((12,2))    

    mCDWthickness = np.zeros(12)
    mCDW_CI = np.zeros((12,2))
    mCDW_yerr = np.zeros((12,2))    

    totalWaterColumnSampled = np.zeros(12)
    
    for i in range(12):
        monthmask = df.JULD.dt.month.isin([i+1]) & df.CTEMP.notnull()
        timeSlice = df.loc[monthmask]

        datalength = len(timeSlice)

        if datalength > 0:
            zlowest = timeSlice.DEPTH.min() 
            number_bins = np.abs(zlowest) // zbin
            zbin_exact = np.abs(zlowest) / float(number_bins)
            depth_bins = np.linspace(zlowest-1, 0, number_bins)

            DSW = (timeSlice.gamman > 28.27) & (timeSlice.PSAL_ADJUSTED > 34.5) & (timeSlice.CTEMP <= -1.8) & (timeSlice.CTEMP >= -1.9)
            lssw = (timeSlice.PSAL_ADJUSTED >= 34.3) & (timeSlice.PSAL_ADJUSTED <= 34.4) & (timeSlice.CTEMP <= -1.5) & (timeSlice.CTEMP > -1.9)
            ISW = (timeSlice.CTEMP < -1.9)
            CDW = (timeSlice.CTEMP >= 0) & (timeSlice.PSAL_ADJUSTED >= 34.5)
            mCDW = (timeSlice.CTEMP < 0) & (timeSlice.CTEMP > -1.8) & (timeSlice.gamman > 28) & (timeSlice.gamman < 28.27)
            
            totalcount = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins) ).CTEMP.count().values
            DSWcount = timeSlice.loc[DSW].groupby(pd.cut(timeSlice.loc[DSW].DEPTH, depth_bins) ).CTEMP.count().values
            LSSWcount = timeSlice.loc[lssw].groupby(pd.cut(timeSlice.loc[lssw].DEPTH, depth_bins) ).CTEMP.count().values
            ISWcount = timeSlice.loc[ISW].groupby(pd.cut(timeSlice.loc[ISW].DEPTH, depth_bins) ).CTEMP.count().values
            mCDWcount = timeSlice.loc[mCDW].groupby(pd.cut(timeSlice.loc[mCDW].DEPTH, depth_bins) ).CTEMP.count().values
            CDWcount = timeSlice.loc[CDW].groupby(pd.cut(timeSlice.loc[CDW].DEPTH, depth_bins) ).CTEMP.count().values
            
            DSWthickness[i] = np.nansum((DSWcount / totalcount) * zbin_exact)
            
            lsswthickness[i] = np.nansum((LSSWcount / totalcount) * zbin_exact)
            
            ISWthickness[i] = np.nansum((ISWcount / totalcount) * zbin_exact)

            CDWthickness[i] = np.nansum((CDWcount / totalcount) * zbin_exact)

            mCDWthickness[i] = np.nansum((mCDWcount / totalcount) * zbin_exact)

            totalWaterColumnSampled[i] = np.nansum(totalcount / totalcount) * zbin_exact
            # computing the Monte Carlo means using repetetive resampling with replacement, with number of repetitions = reps
            # note that if a depth bin has low number of samples, then the resampling method does not add any new information
            MCmeans = np.stack(timeSlice.groupby(pd.cut(timeSlice.DEPTH, np.linspace(zlowest-1, 0, number_bins) ) ).apply(resample_depthbinData_profWise, reps=100).values)
            
            DSW_CI[i] =  np.nanpercentile( np.nansum( (MCmeans[:, 0, :] * zbin), axis=0 ), [0.025, 0.975])
            DSW_yerr[i] = [abs(DSWthickness[i] - DSW_CI[i][0]), abs(DSWthickness[i] - DSW_CI[i][1]) ]
            
            lssw_CI[i] =  np.nanpercentile( np.nansum( (MCmeans[:, 1, :] * zbin), axis=0 ), [0.025, 0.975])
            lssw_yerr[i] = [abs(lsswthickness[i] - lssw_CI[i][0]), abs(lsswthickness[i] - lssw_CI[i][0]) ]
            
            ISW_CI[i] =  np.nanpercentile( np.nansum( (MCmeans[:, 2, :] * zbin), axis=0 ), [0.025, 0.975])
            ISW_yerr[i] = [abs(ISWthickness[i] - ISW_CI[i][0]), abs(ISWthickness[i] - ISW_CI[i][1]) ]
            
            CDW_CI[i] =  np.nanpercentile( np.nansum( (MCmeans[:, 3, :] * zbin), axis=0 ), [0.025, 0.975])
            CDW_yerr[i] = [abs(CDWthickness[i] - CDW_CI[i][0]), abs(CDWthickness[i] - CDW_CI[i][1])]
                        
            mCDW_CI[i] =  np.nanpercentile( np.nansum( (MCmeans[:, 4, :] * zbin), axis=0 ), [0.025, 0.975])
            mCDW_yerr[i] = [abs(mCDWthickness[i] - mCDW_CI[i][0]), abs(mCDWthickness[i] - mCDW_CI[i][1]) ]

        else:
            totalWaterColumnSampled[i] = np.nan
            DSWthickness[i] = np.nan
            lsswthickness[i] = np.nan
            ISWthickness[i] = np.nan
            CDWthickness[i] = np.nan
            mCDWthickness[i] = np.nan
            
    DSWthickness = ma.masked_array(DSWthickness, mask = (DSWthickness == 0) )
    lsswthickness = ma.masked_array(lsswthickness, mask = (lsswthickness == 0) )
    ISWthickness = ma.masked_array(ISWthickness, mask = (ISWthickness == 0) )
    mCDWthickness = ma.masked_array(mCDWthickness, mask = (mCDWthickness == 0) )    
    CDWthickness = ma.masked_array(CDWthickness, mask = (CDWthickness == 0) )
        
    timeaxis = np.arange(1, 13, 1)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axwmb.set_xlim(0, 13)
    axdod.set_xlim(0, 13)
    axdod.set_xticks(timeaxis)
    axdod.xaxis.set_tick_params(width=0.1)
        
    if ymax:
        axwmb.set_ylim(ymin, ymax)
    axdod.set_ylim(0, ymax_dod)
    axwmb.set_xticks(timeaxis)
    axwmb.set_xticklabels(timeaxis_ticklabel)
    if yticks:
        axwmb.set_yticks(yticks)
    if yticks_dod:
        axdod.set_yticks(yticks_dod)

    axwmb.errorbar(timeaxis-2*wd, DSWthickness, yerr=[DSW_yerr.T[0], DSW_yerr.T[1]], label="DSW", zorder=3, color='b', markersize=markersize, capsize=3, fmt="o")
    
    axwmb.errorbar(timeaxis-wd, lsswthickness, yerr=[lssw_yerr.T[0], lssw_yerr.T[1]], label="LSSW", zorder=3, color='aqua', markersize=markersize, capsize=3, fmt="v")
    
    axwmb.errorbar(timeaxis, ISWthickness, yerr=[ISW_yerr.T[0], ISW_yerr.T[1]], label="ISW", zorder=3, color='slategray', markersize=markersize, capsize=3, fmt="^")
    
    axwmb.errorbar(timeaxis+wd, CDWthickness, yerr=[CDW_yerr.T[0], CDW_yerr.T[1]], label="CDW", zorder=3, color='r', markersize=markersize, capsize=3, fmt="X")
    
        
    axwmb.errorbar(timeaxis+2*wd, mCDWthickness, yerr=[mCDW_yerr.T[0], mCDW_yerr.T[1]], label="mCDW", zorder=3, color='darkorange', markersize=markersize, capsize=3, fmt="_")
    
    axdod.scatter(timeaxis, np.abs(totalWaterColumnSampled), color='k', s=3, marker='o', zorder=3)
    
    axwmb.grid(zorder=0, linestyle='dotted')
    axdod.grid(linestyle='dotted', zorder=0)
    axdod.spines["top"].set_visible(False)
    axdod.spines["right"].set_linewidth(0.5)
    axdod.spines["left"].set_linewidth(0.5)
            
    if showlegend:
        axwmb.legend()
    if retValue:
        return np.array(DSWthickness), DSW_CI, np.array(lsswthickness), lssw_CI, np.array(ISWthickness), ISW_CI, np.array(mCDWthickness), mCDW_CI, np.array(CDWthickness), CDW_CI, totalWaterColumnSampled


    
def plot_waterMassThicknessVsMonths(axwmb,axdod, df, ymin=0, ymax=None, yticks=[], zbin=10, wd = 0.1, fontsize=8, showlegend=False, retValue=False, ymax_dod=None, yticks_dod=[]):
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements    

    DSWthickness = np.zeros(12)
    DSWmass = np.zeros(12)
    DSWmassError = np.zeros(12)

    lsswthickness = np.zeros(12)
    lsswmass = np.zeros(12)
    lsswmassError = np.zeros(12)

    ISWthickness = np.zeros(12)
    ISWmass = np.zeros(12)
    ISWmassError = np.zeros(12)

    CDWthickness = np.zeros(12)
    CDWmass = np.zeros(12)
    CDWmassError = np.zeros(12)

    mCDWthickness = np.zeros(12)
    mCDWmass = np.zeros(12)
    mCDWmassError = np.zeros(12)

    zlowest = np.zeros(12)
    
    for i in range(12):
        monthmask = df.JULD.dt.month.isin([i+1])
        timeSlice = df.loc[monthmask]

        salnonull = ~timeSlice.PSAL_ADJUSTED.isnull()
        tempnonull = ~timeSlice.TEMP_ADJUSTED.isnull()
        presnonull = ~timeSlice.PRES_ADJUSTED.isnull()
        datalength = len(timeSlice.loc[salnonull & tempnonull & presnonull])

        if datalength > 0:
            zlowest[i] = timeSlice.loc[salnonull & tempnonull, 'DEPTH'].min() #df.DEPTH.quantile(0.05)
            number_bins = np.abs(zlowest[i]) // zbin
            depth_bins = np.linspace(zlowest[i], 0, number_bins)

            salmean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).PSAL_ADJUSTED.mean().values
            thetamean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).CTEMP.mean().values
            rhomean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).DENSITY_INSITU.mean().values
            rhostd = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).DENSITY_INSITU.std().values
            gammamean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).gamman.mean().values

            DSWbins = ((salmean > 34.4) & (gammamean >= 28.27) & (thetamean <= -1.8) )                  # Williams et al. 2016
            lsswbins = ( (salmean >= 34.3) & (salmean <= 34.4) & (thetamean <= -1.5) & (thetamean > -1.9) )                  # Schodlok et al. 2015
            ISWbins = ( (thetamean < -1.9))                                                             # Williams et al. 2016
            CDWbins = ((salmean >= 34.5) & (thetamean >= 0.0) )                                         # common definition, also named as Warm Deep Water (WDW)
            mCDWbins = ((thetamean > -1.8) & (thetamean < 0) & (gammamean > 28) & (gammamean < 28.27) ) # Williams 2016
            
            DSWthickness[i] = len(DSWbins.nonzero()[0]) * zbin
            DSWmass[i] = zbin * np.nansum(rhomean[DSWbins])
            DSWmassError[i] = np.sqrt(np.nansum(rhostd[DSWbins]**2)) * zbin
            
            lsswthickness[i] = len(lsswbins.nonzero()[0]) * zbin
            lsswmass[i] = zbin * np.nansum(rhomean[lsswbins])
            lsswmassError[i] = np.sqrt(np.nansum(rhostd[lsswbins]**2)) * zbin
            
            ISWthickness[i] = len(ISWbins.nonzero()[0]) * zbin
            ISWmass[i] = zbin * np.nansum(rhomean[ISWbins])
            ISWmassError[i] = np.sqrt(np.nansum(rhostd[ISWbins]**2)) * zbin

            CDWthickness[i] = len(CDWbins.nonzero()[0]) * zbin
            CDWmass[i] = zbin * np.nansum(rhomean[CDWbins])
            CDWmassError[i] = np.sqrt(np.nansum(rhostd[CDWbins]**2)) * zbin

            mCDWthickness[i] = len(mCDWbins.nonzero()[0]) * zbin
            mCDWmass[i] = zbin * np.nansum(rhomean[mCDWbins])
            mCDWmassError[i] = np.sqrt(np.nansum(rhostd[mCDWbins]**2)) * zbin


        else:
            zlowest[i] = np.nan
            DSWthickness[i] = np.nan
            lsswthickness[i] = np.nan
            ISWthickness[i] = np.nan
            CDWthickness[i] = np.nan
            mCDWthickness[i] = np.nan

    timeaxis = np.arange(1, 13, 1)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axwmb.set_xlim(0, 13)
    axdod.set_xlim(0, 13)
    axdod.set_xticks(timeaxis)
    axdod.xaxis.set_tick_params(width=0.1)
    #axdod.yaxis.set_tick_params(width=5)
    
    if ymax:
        axwmb.set_ylim(ymin, ymax)
    axdod.set_ylim(0, ymax_dod)
    axwmb.set_xticks(timeaxis)
    axwmb.set_xticklabels(timeaxis_ticklabel)
    if yticks:
        axwmb.set_yticks(yticks)
    if yticks_dod:
        axdod.set_yticks(yticks_dod)

    axwmb.bar(timeaxis-2*wd, DSWthickness, wd, label="DSW", zorder=3, color='b')
    #ax.errorbar(timeaxis-2*wd, DSWmass, yerr=DSWmassError, fmt='o', markersize=1, capsize=3, color='k')

    axwmb.bar(timeaxis-wd, lsswthickness,wd, label="LSSW", zorder=3, color='g')
    axwmb.bar(timeaxis, ISWthickness, wd, label="ISW", zorder=3, color='slategray')
    #ax.axhline(y= np.abs(depth_5quantile), linestyle='dashed', color='b', zorder=3)
    axwmb.bar(timeaxis+wd, CDWthickness, wd, label="CDW", zorder=3, color='r')
    #ax.errorbar(timeaxis+wd, CDWmass, yerr=CDWmassError, fmt='o', markersize=1, capsize=3, color='k')
        
    axwmb.bar(timeaxis+2*wd, mCDWthickness, wd, label="mCDW", zorder=3, color='darkorange')
    axdod.scatter(timeaxis, np.abs(zlowest), color='k', s=3, marker='o', zorder=3)
    axwmb.grid(zorder=0, linestyle='dotted')
    axdod.grid(linestyle='dotted', zorder=0)
    axdod.spines["top"].set_visible(False)
    axdod.spines["right"].set_linewidth(0.5)
    axdod.spines["left"].set_linewidth(0.5)
            
    if showlegend:
        axwmb.legend()
    if retValue:
        return DSWthickness, lsswthickness, ISWthickness, mCDWthickness, CDWthickness, zlowest


###################################################################
###################################################################
############### BOOTSTRAPPER FUNCTION #############################
def resample_depthbinData(timeDepthSliced, reps=1000):
    sliceArray = timeDepthSliced.loc[:, ["PSAL_ADJUSTED", "CTEMP", "gamman"] ].values
    try:
        randidxr = np.random.choice(len(sliceArray), (len(sliceArray), reps), replace=True )
        return np.nanmean(sliceArray[randidxr] , axis=0)
    except:
        return np.full_like(np.zeros((reps, 3)), np.nan)
###################################################################
###################################################################


def waterMassThickness_bootstrapper(axwmb,axdod, df, ymin=0, ymax=None, yticks=[], zbin=20, wd = 0.1, fontsize=8, showlegend=False, retValue=False, ymax_dod=None, yticks_dod=[], reps=100, markersize=3):
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements    

    DSWthickness = np.zeros(12)
    DSW_CI = np.zeros((12,2))
    DSW_yerr = np.zeros((12,2))    

    lsswthickness = np.zeros(12)
    lssw_CI = np.zeros((12,2))
    lssw_yerr = np.zeros((12,2))    

    ISWthickness = np.zeros(12)
    ISW_CI = np.zeros((12,2))
    ISW_yerr = np.zeros((12,2))    

    CDWthickness = np.zeros(12)
    CDW_CI = np.zeros((12,2))
    CDW_yerr = np.zeros((12,2))    

    mCDWthickness = np.zeros(12)
    mCDW_CI = np.zeros((12,2))
    mCDW_yerr = np.zeros((12,2))    

    totalWaterColumnSampled = np.zeros(12)
    
    for i in range(12):
        monthmask = df.JULD.dt.month.isin([i+1])
        timeSlice = df.loc[monthmask]

        salnonull = ~timeSlice.PSAL_ADJUSTED.isnull()
        tempnonull = ~timeSlice.TEMP_ADJUSTED.isnull()
        presnonull = ~timeSlice.PRES_ADJUSTED.isnull()
        ctempnonull = ~timeSlice.CTEMP.isnull()
        datalength = len(timeSlice.loc[salnonull & tempnonull & presnonull])

        if datalength > 0:
            zlowest = timeSlice.loc[salnonull & tempnonull & presnonull, 'DEPTH'].min() #df.DEPTH.quantile(0.05)
            number_bins = np.abs(zlowest) // zbin
            zbin_exact = np.abs(zlowest) / float(number_bins)
            depth_bins = np.linspace(zlowest-1, 0, number_bins)
            totalWaterColumnSampled[i] = len(timeSlice.loc[ctempnonull].groupby(pd.cut(timeSlice.loc[ctempnonull].DEPTH, depth_bins ) ).CTEMP.count().nonzero()[0]) * zbin_exact
            
            salmean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).PSAL_ADJUSTED.mean().values
            thetamean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).CTEMP.mean().values
            rhomean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).DENSITY_INSITU.mean().values
            rhostd = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).DENSITY_INSITU.std().values
            gammamean = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins)).gamman.mean().values

            DSWbins = ((salmean > 34.5) & (gammamean >= 28.27) & (thetamean <= -1.8) ) & (thetamean >= -1.9)                  # Williams et al. 2016
            lsswbins = ( (salmean >= 34.3) & (salmean <= 34.4) & (thetamean <= -1.5) & (thetamean > -1.9) )                  # Schodlok et al. 2015
            ISWbins = ( (thetamean < -1.9))                                                             # Williams et al. 2016
            CDWbins = ((salmean >= 34.5) & (thetamean >= 0.0) )                                         # common definition, also named as Warm Deep Water (WDW)
            mCDWbins = ((thetamean > -1.8) & (thetamean < 0) & (gammamean > 28) & (gammamean < 28.27) ) # Williams 2016
            
            DSWthickness[i] = len(DSWbins.nonzero()[0]) * zbin_exact
            
            lsswthickness[i] = len(lsswbins.nonzero()[0]) * zbin_exact
            
            ISWthickness[i] = len(ISWbins.nonzero()[0]) * zbin_exact

            CDWthickness[i] = len(CDWbins.nonzero()[0]) * zbin_exact

            mCDWthickness[i] = len(mCDWbins.nonzero()[0]) * zbin_exact

            # computing the Monte Carlo means using repetetive resampling with replacement, with number of repetitions = reps
            # note that if a depth bin has low number of samples, then the resampling method does not add any new information
            MCmeans = np.stack(timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins ) ).apply(resample_depthbinData, reps=reps).values)
            
            DSWbool = ((MCmeans[:, :, 0] > 34.5) & (MCmeans[:,:, 1] <= -1.8) & (MCmeans[:,:, 1] >= -1.9) & (MCmeans[:,:,2] >=28.27))
            DSWbootstrapped = np.nansum(DSWbool, axis=0) * zbin_exact
            DSWdelta = DSWbootstrapped - DSWthickness[i]
            DSW_CI[i] =  DSWthickness[i] - np.percentile(np.sort(DSWdelta), [2.5, 97.5])[::-1]
            DSW_yerr[i] = [abs(DSWthickness[i] - DSW_CI[i][0]), abs(DSWthickness[i] - DSW_CI[i][1]) ]
            
            lsswbool = ( (MCmeans[:,:,0] >= 34.3) & (MCmeans[:,:,0] <= 34.4) & (MCmeans[:,:,1] <= -1.5) & (MCmeans[:,:,1] > -1.9) )                  # Schodlok et al. 2015
            lsswbootstrapped = np.nansum(lsswbool, axis=0) * zbin_exact
            lsswdelta = lsswbootstrapped - lsswthickness[i]
            lssw_CI[i] =  lsswthickness[i] - np.percentile(np.sort(lsswdelta), [2.5, 97.5])[::-1]
            lssw_yerr[i] = [abs(lsswthickness[i] - lssw_CI[i][0]), abs(lsswthickness[i] - lssw_CI[i][0]) ]
            
            ISWbool = ( (MCmeans[:,:,1] < -1.9))                                                             # Williams et al. 2016
            ISWbootstrapped = np.nansum(ISWbool, axis=0) * zbin_exact
            ISWdelta = ISWbootstrapped - ISWthickness[i]
            ISW_CI[i] =  ISWthickness[i] - np.percentile(np.sort(ISWdelta), [2.5, 97.5])[::-1]
            ISW_yerr[i] = [abs(ISWthickness[i] - ISW_CI[i][0]), abs(ISWthickness[i] - ISW_CI[i][1]) ]
            
            CDWbool = ((MCmeans[:,:,0] >= 34.5) & (MCmeans[:,:,1] >= 0.0) )                                         # common definition, also named as Warm Deep Water (WDW)
            CDWbootstrapped = np.nansum(CDWbool, axis=0) * zbin_exact
            CDWdelta = CDWbootstrapped - CDWthickness[i]
            CDW_CI[i] =  CDWthickness[i] - np.percentile(np.sort(CDWdelta), [2.5, 97.5])[::-1]
            CDW_yerr[i] = [abs(CDWthickness[i] - CDW_CI[i][0]), abs(CDWthickness[i] - CDW_CI[i][1])]
                        
            mCDWbool = ((MCmeans[:,:,1] > -1.8) & (MCmeans[:,:,1] < 0) & (MCmeans[:,:,2] > 28) & (MCmeans[:,:,2] < 28.27) ) # Williams 2016
            mCDWbootstrapped = np.nansum(mCDWbool, axis=0) * zbin_exact
            mCDWdelta = mCDWbootstrapped - mCDWthickness[i]
            mCDW_CI[i] =  mCDWthickness[i] - np.percentile(np.sort(mCDWdelta), [2.5, 97.5])[::-1]
            mCDW_yerr[i] = [abs(mCDWthickness[i] - mCDW_CI[i][0]), abs(mCDWthickness[i] - mCDW_CI[i][1]) ]

        else:
            totalWaterColumnSampled[i] = np.nan
            DSWthickness[i] = np.nan
            lsswthickness[i] = np.nan
            ISWthickness[i] = np.nan
            CDWthickness[i] = np.nan
            mCDWthickness[i] = np.nan
            
    DSWthickness = ma.masked_array(DSWthickness, mask = (DSWthickness == 0) )
    lsswthickness = ma.masked_array(lsswthickness, mask = (lsswthickness == 0) )
    ISWthickness = ma.masked_array(ISWthickness, mask = (ISWthickness == 0) )
    mCDWthickness = ma.masked_array(mCDWthickness, mask = (mCDWthickness == 0) )    
    CDWthickness = ma.masked_array(CDWthickness, mask = (CDWthickness == 0) )
        
    timeaxis = np.arange(1, 13, 1)
    timeaxis_ticklabel = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axwmb.set_xlim(0, 13)
    axdod.set_xlim(0, 13)
    axdod.set_xticks(timeaxis)
    axdod.xaxis.set_tick_params(width=0.1)
        
    if ymax:
        axwmb.set_ylim(ymin, ymax)
    axdod.set_ylim(0, ymax_dod)
    axwmb.set_xticks(timeaxis)
    axwmb.set_xticklabels(timeaxis_ticklabel)
    if yticks:
        axwmb.set_yticks(yticks)
    if yticks_dod:
        axdod.set_yticks(yticks_dod)

    axwmb.errorbar(timeaxis-2*wd, DSWthickness, yerr=[DSW_yerr.T[0], DSW_yerr.T[1]], label="DSW", zorder=3, color='b', markersize=markersize, capsize=3, fmt="o")
    
    axwmb.errorbar(timeaxis-wd, lsswthickness, yerr=[lssw_yerr.T[0], lssw_yerr.T[1]], label="LSSW", zorder=3, color='aqua', markersize=markersize, capsize=3, fmt="v")
    
    axwmb.errorbar(timeaxis, ISWthickness, yerr=[ISW_yerr.T[0], ISW_yerr.T[1]], label="ISW", zorder=3, color='slategray', markersize=markersize, capsize=3, fmt="^")
    
    axwmb.errorbar(timeaxis+wd, CDWthickness, yerr=[CDW_yerr.T[0], CDW_yerr.T[1]], label="CDW", zorder=3, color='r', markersize=markersize, capsize=3, fmt="X")
    
    axwmb.errorbar(timeaxis+2*wd, mCDWthickness, yerr=[mCDW_yerr.T[0], mCDW_yerr.T[1]], label="mCDW", zorder=3, color='darkorange', markersize=markersize, capsize=3, fmt="_")
    
    axdod.scatter(timeaxis, np.abs(totalWaterColumnSampled), color='k', s=3, marker='o', zorder=3)
    
    axwmb.grid(zorder=0, linestyle='dotted')
    axdod.grid(linestyle='dotted', zorder=0)
    axdod.spines["top"].set_visible(False)
    axdod.spines["right"].set_linewidth(0.5)
    axdod.spines["left"].set_linewidth(0.5)
            
    if showlegend:
        axwmb.legend()
    if retValue:
        return np.array(DSWthickness), DSW_CI, np.array(lsswthickness), lssw_CI, np.array(ISWthickness), ISW_CI, np.array(mCDWthickness), mCDW_CI, np.array(CDWthickness), CDW_CI, totalWaterColumnSampled



def plot_array_waterMassThickness(df, regions, titles, ymin=0, ymax=None, bar_width=0.15, wd=7.48, ht=7., mrows=13, retValue=True, yticks_dod=[], ymax_dod=None, plotter = 1,
                                  ncols=3, height_ratios=[], width_ratios=[], save=False, savename="savedfig.png", show=True, fontsize=8, zbin=20.0, yticks=[], reps=1000, hspace=0, wspace=0):
    
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
    if not height_ratios:
        height_ratios = [0.5,1, 0.05, 0.5,1, 0.25, 0.5,1, 0.05, 0.5,1, 0.25, 0.25] 
    if not width_ratios:
        width_ratios = [1]*ncols
        
    gs = gridspec.GridSpec(mrows, ncols, height_ratios=height_ratios, width_ratios=width_ratios)
    gs.update(hspace= hspace)
    gs.update(wspace= wspace)
    
    count = 0
    DSWthickness  = np.zeros((len(titles), 12))
    DSW_CI = np.zeros((len(titles), 12, 2) )
    lsswthickness = np.zeros((len(titles), 12))
    lssw_CI = np.zeros((len(titles), 12, 2) )
    ISWthickness = np.zeros((len(titles), 12))
    ISW_CI = np.zeros((len(titles), 12, 2) )
    mCDWthickness = np.zeros((len(titles), 12))
    mCDW_CI = np.zeros((len(titles), 12, 2) )
    CDWthickness = np.zeros((len(titles), 12))
    CDW_CI = np.zeros((len(titles), 12, 2) )
    zlowest = np.zeros((len(titles), 12) )
    
    wmbind = [1,4,7,10]
    for i in wmbind:
        for j in range(ncols):
            axwmb = plt.subplot(gs[i,j]) # water mass budget axis
            axdod = plt.subplot(gs[i-1, j]) # depth of dive axis

            if retValue:
                if(plotter == 1):
                    DSWthickness[count], DSW_CI[count], lsswthickness[count], lssw_CI[count], ISWthickness[count], ISW_CI[count], mCDWthickness[count], mCDW_CI[count], CDWthickness[count], CDW_CI[count], zlowest[count] = waterMassThickness_bootstrapper(axwmb,axdod, df.loc[regions[count]], wd=bar_width, yticks=yticks, ymax=ymax, retValue=retValue, yticks_dod=yticks_dod, ymax_dod=ymax_dod, zbin=zbin, reps=reps)
                elif(plotter == 2):
                    DSWthickness[count], DSW_CI[count], lsswthickness[count], lssw_CI[count], ISWthickness[count], ISW_CI[count], mCDWthickness[count], mCDW_CI[count], CDWthickness[count], CDW_CI[count], zlowest[count] = waterMassThickness_bootstrapper_profileWise(axwmb,axdod, df.loc[regions[count]], wd=bar_width, yticks=yticks, ymax=ymax, retValue=retValue, yticks_dod=yticks_dod, ymax_dod=ymax_dod, zbin=zbin, reps=reps)
                else:
                    raise ValueError('plotter can only have value 1 or 2')
            else:
                if(plotter == 1):
                    waterMassThickness_bootstrapper(axwmb,axdod, df.loc[regions[count]], wd=bar_width, yticks=yticks, ymax=ymax, retValue=retValue, yticks_dod=yticks_dod, ymax_dod=ymax_dod, zbin=zbin, reps=reps)
                elif(plotter == 2):
                    waterMassThickness_bootstrapper_profileWise(axwmb,axdod, df.loc[regions[count]], wd=bar_width, yticks=yticks, ymax=ymax, retValue=retValue, yticks_dod=yticks_dod, ymax_dod=ymax_dod, zbin=zbin, reps=reps)
                else:
                    raise ValueError('plotter can only have value 1 or 2')

                
            
            axdod.set_ylabel("m")

            axwmb.set_ylabel("thickness (m)")
            if((j > 0)):
                axwmb.set_yticklabels([])
                axwmb.set_ylabel("")
                axwmb.yaxis.set_tick_params(width=0.1)
                axdod.set_yticklabels([])
                axdod.set_ylabel("")
                axdod.yaxis.set_tick_params(width=0.1)
                axdod.spines["left"].set_linewidth(0.5)
                
            if( ((i == 1) or (i == 7)) ):
                axwmb.set_xticklabels([])
                axwmb.set_xlabel("")
            axdod.set_xticklabels([])
            axdod.set_xlabel("")

            if( (titles[count] == "(l) BS-DIS") ): # or (titles[count] == "(e) PHCA2") ):
                axwmb.set_ylim(0,1500)
                yticksBSA2 = [250, 750, 1250]
                axwmb.set_yticks(yticksBSA2)
                axwmb.set_yticklabels(yticksBSA2)
                axwmb.yaxis.set_tick_params(width=1)
                axwmb.annotate(s=titles[count], xy=(0.5,1050))
            else:
                axwmb.annotate(s=titles[count], xy=(0.5,700))

                
            count += 1
            if(count == len(regions)):
                break
        if(count == len(regions)):
            break

    handles, labels = axwmb.get_legend_handles_labels()
    axlegend = plt.subplot(gs[12, :])
    axlegend.legend(handles, labels, ncol=5, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)
    
    #plt.tight_layout()
    
    if save:
        plt.savefig(savename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    if retValue:
        return DSWthickness, DSW_CI, lsswthickness, lssw_CI, ISWthickness, ISW_CI,  mCDWthickness, mCDW_CI, CDWthickness, CDW_CI, zlowest
    ####        0              1         2            3         4             5          6           7          8           9       10

    
def plot_WaterMass_Correlation(wmCorr, regionsName=[], titles=[], wd=7.48, ht=9, mrows=4, ncols=3, save=False, savename="untitled.png",
                               windtypes=['neg_u10SlopeMonMean', 'v10SlopeMonMean', 'neg_stress_curl']):
    matplotlib.rcParams.update({'font.size': 8})        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
        
    gs = gridspec.GridSpec(mrows+1, ncols, height_ratios=[1]*mrows+[0.1], width_ratios=[1]*ncols)
    gs.update(hspace=0.4)
    #gs.update(wspace=0.5)

    wth = 0.1
    axarr = []
    count = 0
    for i in range(mrows):
        for j in range(ncols):
            axarr.append(plt.subplot(gs[i,j]))
            
            for k in range(len(windtypes)):
                if(k == 0):
                    labels = ["CDW", "mCDW", "LSSW", "DSW", "ISW"]
                else:
                    labels=[None]*5

                axarr[-1].bar( (k+1)-1.5*wth, wmCorr.loc[regionsName[count], windtypes[k] ][0], wth, label=labels[0], color="r")
                axarr[-1].bar( (k+1)-.5*wth, wmCorr.loc[regionsName[count], windtypes[k] ][4], wth, label=labels[1], color="darkorange")
                axarr[-1].bar( (k+1)+.5*wth, wmCorr.loc[regionsName[count], windtypes[k] ][3], wth, label=labels[2], color="g")
                axarr[-1].bar( (k+1)+1.5*wth, wmCorr.loc[regionsName[count], windtypes[k] ][1], wth, label=labels[3], color="b")
                axarr[-1].bar( (k+1)+1.5*wth, wmCorr.loc[regionsName[count], windtypes[k] ][2], wth, label=labels[4], color="slategray")
            axarr[-1].axhline(y=0)
            axarr[-1].set_title(titles[count])
            if(i == mrows-1):
                axarr[-1].set_xticklabels([None]+["-U$_{Slope}$", "V$_{Slope}$", "$-(\\nabla \\times \\tau)$"], rotation=0)
            else:
                axarr[-1].set_xticklabels([])
                axarr[-1].set_xlabel("")
            
            if((i == mrows-1) & (j == 1) ):
                axarr[-1].set_xlabel("Wind types")
            if((j == 0)):
                axarr[-1].set_ylabel("Correlation")
            count+=1
    
    handles, labels = axarr[-1].get_legend_handles_labels()
    axlegend = plt.subplot(gs[mrows, :])
    axlegend.legend(handles, labels, ncol=4, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    if save:
        plt.savefig(savename, dpi=600)
    plt.show()


def plot_WaterMass_Correlation_byRegime(regimeCorrArr, titles=[], wd=7.48, ht=6, mrows=2, ncols=4, save=False, savename="untitled.png",
                               windtypes=['neg_u10SlopeMonMean', 'v10SlopeMonMean', 'neg_stress_curl']):
    matplotlib.rcParams.update({'font.size': 8})        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
        
    gs = gridspec.GridSpec(mrows+1, ncols, height_ratios=[1]*mrows+[0.1], width_ratios=[1]*ncols)
    gs.update(hspace=0.7)
    gs.update(wspace=0.05)

    wth = 0.1
    axarr = []
    count = 0
    for i in range(mrows):
        for j in range(ncols):
            axarr.append(plt.subplot(gs[i,j]))

            for k in range(len(windtypes)):
                if(k == 0):
                    labels = ["CDW", "mCDW", "LSSW", "DSW", "ISW"]
                else:
                    labels = [None]*5

                axarr[-1].bar( float(k+1)-2.*wth, regimeCorrArr[count].loc["CDW", windtypes[k] ], wth, label=labels[0], color="r")
                axarr[-1].bar( (k+1)-1.*wth, regimeCorrArr[count].loc["mCDW", windtypes[k] ], wth, label=labels[1], color="darkorange")
                axarr[-1].bar( (k+1)+0*wth, regimeCorrArr[count].loc["LSSW", windtypes[k] ], wth, label=labels[2], color="g")
                axarr[-1].bar( (k+1)+1.*wth, regimeCorrArr[count].loc["DSW", windtypes[k] ], wth, label=labels[3], color="b")
                axarr[-1].bar( (k+1)+2.*wth, regimeCorrArr[count].loc["ISW", windtypes[k] ], wth, label=labels[4], color="slategray")

            ## count_ax = axarr[-1].twiny()
            ## ax_xlim = axarr[-1].get_xlim()
            ## count_ax.set_xlim(ax_xlim)
            ## xticks = np.array([1]*5 + [2]*5 + [3]*5) + np.array([-2*wth,-wth,0,+wth,+2*wth]*3)
            ## count_ax.set_xticks(xticks)
            ## count_ax.set_xticklabels(salmonth_yearcount)
            axarr[-1].tick_params(axis="x", direction="in")
            axarr[-1].axhline(y=0)
            axarr[-1].set_title(titles[count])
            axarr[-1].set_ylim(-1, 1)
            #if(i == mrows-1):
            axarr[-1].set_xticklabels([None]+["-U$_{Slope}$", "V$_{Slope}$", "-$( \\vec{\\nabla} \\times \\vec{\\tau} )$", "W$_{shelf}$"], rotation=0)
            ## else:
            ##     axarr[-1].set_xticklabels([])
            ##     axarr[-1].set_xlabel("")
            axarr[-1].set_yticks(np.arange(-1, 1.1, 0.25))
            if((i == mrows-1) ):
                axarr[-1].set_xlabel("Wind forcing")
            if((j == 0)):
                axarr[-1].set_ylabel("Correlation")
            else:
                axarr[-1].set_yticklabels([])
                axarr[-1].set_ylabel("")
            axarr[-1].grid(linestyle=":")    
            count+=1
    
    handles, labels = axarr[-1].get_legend_handles_labels()
    axlegend = plt.subplot(gs[mrows, :])
    axlegend.legend(handles, labels, ncol=3, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    if save:
        plt.savefig(savename, dpi=600)
    plt.show()


def plot_WaterMass_Correlation_byRegime2(regimeCorrArr, waterMassThickness, titles=[], wd=7.48, ht=7, save=False, savename="untitled.png",
                               windtypes=['neg_u10SlopeMonMean', 'v10SlopeMonMean', 'neg_stress_curl']):
    matplotlib.rcParams.update({'font.size': 8})        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
        
    gs = gridspec.GridSpec(7, 8, height_ratios=[1, 0.5, 1, 0.5, 1, 0.25, 0.025], width_ratios=[1]*8)
    gs.update(hspace=0.)
    gs.update(wspace=0.0)

    wth = 0.1
    axarr = []
    count = 0
    for i in range(0,3,2):
        for j in range(0,7,2):
            axarr.append(plt.subplot(gs[i,j:j+2]))

            for k in range(len(windtypes)):
                if(k == 0):
                    labels = ["CDW", "mCDW", "LSSW", "DSW", "ISW"]
                else:
                    labels = [None]*5
                
                axarr[-1].bar( float(k+1)-2.*wth, regimeCorrArr[count].loc["CDW", windtypes[k] ], wth, label=labels[0], color="r")
                axarr[-1].bar( (k+1)-1.*wth, regimeCorrArr[count].loc["mCDW", windtypes[k] ], wth, label=labels[1], color="darkorange")
                axarr[-1].bar( (k+1)+0*wth, regimeCorrArr[count].loc["LSSW", windtypes[k] ], wth, label=labels[2], color="g")
                axarr[-1].bar( (k+1)+1.*wth, regimeCorrArr[count].loc["DSW", windtypes[k] ], wth, label=labels[3], color="b")
                axarr[-1].bar( (k+1)+2.*wth, regimeCorrArr[count].loc["ISW", windtypes[k] ], wth, label=labels[4], color="slategray")
                axarr[-1].set_xlim(0.5, 3.5)
                axarr[-1].set_xticks([1,2,3])
            ## count_ax = axarr[-1].twiny()
            ## ax_xlim = axarr[-1].get_xlim()
            ## count_ax.set_xlim(ax_xlim)
            ## xticks = np.array([1]*5 + [2]*5 + [3]*5) + np.array([-2*wth,-wth,0,+wth,+2*wth]*3)
            ## count_ax.set_xticks(xticks)
            ## count_ax.set_xticklabels(salmonth_yearcount)

            axarr[-1].axhline(y=0)
            axarr[-1].set_title(titles[count])
            axarr[-1].set_ylim(-0.8, 0.8)
            #if(i == mrows-1):
            axarr[-1].tick_params(axis="x", direction="in")
            axarr[-1].set_xticklabels(["-U$_{Slope}$", "V$_{Slope}$", "-$( \\vec{\\nabla} \\times \\vec{\\tau} )$", "W$_{shelf}$"], rotation=0)
            ## else:
            ##     axarr[-1].set_xticklabels([])
            ##     axarr[-1].set_xlabel("")
            axarr[-1].set_yticks(np.arange(-0.75, 0.76, 0.25))
            if((i == 1) ):
                axarr[-1].set_xlabel("Wind forcing")
            if((j == 0)):
                axarr[-1].set_ylabel("Correlation")
            else:
                axarr[-1].set_yticklabels([])
                axarr[-1].set_ylabel("")
            axarr[-1].grid(linestyle=":")    
            count+=1
    
    handles, labels = axarr[-1].get_legend_handles_labels()
    
    ## # plotting shelf width correlations
    axarr.append(plt.subplot(gs[4, 3:5]))
    shelfCorrCount = 0
    for area in ["A1", "A2"]:
        wmAreaMask = waterMassThickness.region.str.contains(area)

        axarr[-1].bar( (shelfCorrCount+1)-2.*wth, waterMassThickness.loc[wmAreaMask, "CDW"].corr(waterMassThickness.loc[wmAreaMask, "ShelfWidth"]), wth, label=labels[0], color="r")
        axarr[-1].bar( (shelfCorrCount+1)-1.*wth, waterMassThickness.loc[wmAreaMask, "mCDW"].corr(waterMassThickness.loc[wmAreaMask, "ShelfWidth"]), wth, label=labels[1], color="darkorange")
        axarr[-1].bar( (shelfCorrCount+1)+0*wth, waterMassThickness.loc[wmAreaMask,  "LSSW"].corr(waterMassThickness.loc[wmAreaMask, "ShelfWidth"]), wth, label=labels[2], color="g")
        axarr[-1].bar( (shelfCorrCount+1)+1.*wth, waterMassThickness.loc[wmAreaMask, "DSW"].corr(waterMassThickness.loc[wmAreaMask, "ShelfWidth"]), wth, label=labels[3], color="b")
        axarr[-1].bar( (shelfCorrCount+1)+2.*wth, waterMassThickness.loc[wmAreaMask, "ISW"].corr(waterMassThickness.loc[wmAreaMask, "ShelfWidth"]), wth, label=labels[4], color="slategray")
        shelfCorrCount += 1

        axarr[-1].axhline(y=0)
        axarr[-1].set_title("(i) Correlation\n with shelf width")
        axarr[-1].set_ylim(-0.8, 0.8)
        axarr[-1].set_xlim(0.5, 2.5)
        axarr[-1].set_xticks([1,2])
        axarr[-1].set_xticklabels(["A1", "A2"], rotation=0)
        axarr[-1].set_yticks(np.arange(-0.75, 0.76, 0.25))
        axarr[-1].grid(linestyle=":")    
        

    # creating legend
    axlegend = plt.subplot(gs[6, :])
    axlegend.legend(handles, labels, ncol=5, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)


    if save:
        plt.savefig(savename, dpi=600)
    plt.show()


## Function written on 19th October, to include the Confidence Interval on the pearson's coefficient
def plot_WaterMass_Correlation_byRegime_CI(waterMassThickness, wd=7.48, ht=6, mrows=2, ncols=4, save=False, savename="untitled.png",
                               windtypes=['neg_u10SlopeMonMean', 'v10SlopeMonMean', 'neg_stress_curl'], maskInsignificant=False):
    matplotlib.rcParams.update({'font.size': 8})        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
        
    gs = gridspec.GridSpec(mrows+1, ncols, height_ratios=[1]*mrows+[0.1], width_ratios=[1]*ncols)
    gs.update(hspace=0.7)
    gs.update(wspace=0.05)

    wth = 0.1
    axarr = []
    count = 0

    coldRegimeWideShelfRegionsA1 = ["WSA1", "WPBA1", "EPBA1", "RSA1"]
    coldRegimeWideShelfRegionsA2 = ["WSA2", "WPBA2", "EPBA2", "RSA2"]

    coldRegimeNarrowShelfRegionsA1 = ["CDA1", "ACA1"]
    coldRegimeNarrowShelfRegionsA2 = ["CDA2", "ACA2"]

    lsswRegimeRegionsA1 = ["PMCA1", "LACA1", "KCA1"]
    lsswRegimeRegionsA2 = ["PMCA2", "LACA2", "KCA2"]

    warmRegimeRegionsA1 = ["PHCA1", "ASA1", "BSA1"]
    warmRegimeRegionsA2 = ["PHCA2", "ASA2", "BSA2"]
    
    wmColdRegimeWideShelfMaskA1 = waterMassThickness.region.isin(coldRegimeWideShelfRegionsA1)
    coldRegime_WideShelfCorrA1 = waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeWideShelfRegionsA1)].corr()
    coldRegime_WideShelfCorrA1_yerr = calculate_pvalues(waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeWideShelfRegionsA1)].dropna(), maskInsignificant=maskInsignificant)

    wmColdRegimeWideShelfMaskA2 = waterMassThickness.region.isin(coldRegimeWideShelfRegionsA2)
    coldRegime_WideShelfCorrA2 = waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeWideShelfRegionsA2)].corr()
    coldRegime_WideShelfCorrA2_yerr = calculate_pvalues(waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeWideShelfRegionsA2)].dropna(), maskInsignificant=maskInsignificant)

    wmColdRegimeNarrowShelfMaskA1 = waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA1)
    coldRegime_NarrowShelfCorrA1 = waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA1)].corr()
    coldRegime_NarrowShelfCorrA1_yerr = calculate_pvalues(waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA1)].dropna(), maskInsignificant=maskInsignificant)

    wmColdRegimeNarrowShelfMaskA2 = waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA2)
    coldRegime_NarrowShelfCorrA2 = waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA2)].corr()
    coldRegime_NarrowShelfCorrA2_yerr = calculate_pvalues(waterMassThickness.loc[waterMassThickness.region.isin(coldRegimeNarrowShelfRegionsA2)].dropna(), maskInsignificant=maskInsignificant)

    wmLSSWRegimeMaskA1 = waterMassThickness.region.isin(lsswRegimeRegionsA1)
    lsswRegimeCorrA1 = waterMassThickness.loc[wmLSSWRegimeMaskA1].corr()
    lsswRegimeCorrA1_yerr = calculate_pvalues(waterMassThickness.loc[wmLSSWRegimeMaskA1].dropna(), maskInsignificant=maskInsignificant)
    
    wmLSSWRegimeMaskA2 = waterMassThickness.region.isin(lsswRegimeRegionsA2)
    lsswRegimeCorrA2= waterMassThickness.loc[wmLSSWRegimeMaskA2].corr()
    lsswRegimeCorrA2_yerr = calculate_pvalues(waterMassThickness.loc[wmLSSWRegimeMaskA2].dropna(), maskInsignificant=maskInsignificant)

    wmWarmRegimeMaskA1 = waterMassThickness.region.isin(warmRegimeRegionsA1)
    warmRegimeCorrA1 = waterMassThickness.loc[wmWarmRegimeMaskA1].corr()
    warmRegimeCorrA1_yerr = calculate_pvalues(waterMassThickness.loc[wmWarmRegimeMaskA1].dropna(), maskInsignificant=maskInsignificant)

    wmWarmRegimeMaskA2 = waterMassThickness.region.isin(warmRegimeRegionsA2)
    warmRegimeCorrA2 = waterMassThickness.loc[wmWarmRegimeMaskA2].corr()
    warmRegimeCorrA2_yerr = calculate_pvalues(waterMassThickness.loc[wmWarmRegimeMaskA2].dropna(), maskInsignificant=maskInsignificant)

    regimeCorrArr = [coldRegime_WideShelfCorrA1, coldRegime_NarrowShelfCorrA1, lsswRegimeCorrA1, warmRegimeCorrA1, coldRegime_WideShelfCorrA2, coldRegime_NarrowShelfCorrA2, lsswRegimeCorrA2, warmRegimeCorrA2]
    regimeCorrArr_yerr = [coldRegime_WideShelfCorrA1_yerr, coldRegime_NarrowShelfCorrA1_yerr, lsswRegimeCorrA1_yerr, warmRegimeCorrA1_yerr, coldRegime_WideShelfCorrA2_yerr, coldRegime_NarrowShelfCorrA2_yerr, lsswRegimeCorrA2_yerr, warmRegimeCorrA2_yerr]
    titles = ["(a) Cold Regime \n wide-shelf A1", "(c) Cold Regime \n narrow-shelf A1", "(e) Intermediate \n Regime A1", "(g) Warm Regime A1\n",
              "(b) Cold Regime \n wide-shelf A2", "(d) Cold Regime \n narrow-shelf A2", "(f) Intermediate \n Regime A2", "(h) Warm Regime A2\n"]    
    for i in range(mrows):
        for j in range(ncols):
            axarr.append(plt.subplot(gs[i,j]))

            for k in range(len(windtypes)):
                if(k == 0):
                    labels = ["CDW", "mCDW", "LSSW", "DSW", "ISW"]
                else:
                    labels = [None]*5

                axarr[-1].errorbar( float(k+1)-2.*wth, regimeCorrArr[count].loc["CDW", windtypes[k] ], yerr= np.array([regimeCorrArr_yerr[count].loc["CDW", windtypes[k] ]]).T, label=labels[0], color="r", fmt="x", capsize=3, markersize=3)
                axarr[-1].errorbar( (k+1)-1.*wth, regimeCorrArr[count].loc["mCDW", windtypes[k] ], yerr=np.array([regimeCorrArr_yerr[count].loc["mCDW", windtypes[k] ]]).T, label=labels[1], color="darkorange", fmt="_", capsize=3, markersize=3)
                axarr[-1].errorbar( (k+1)+0*wth, regimeCorrArr[count].loc["LSSW", windtypes[k] ], yerr=np.array([regimeCorrArr_yerr[count].loc["LSSW", windtypes[k] ]]).T, label=labels[2], color="aqua", fmt="v", capsize=3, markersize=3)
                axarr[-1].errorbar( (k+1)+1.*wth, regimeCorrArr[count].loc["DSW", windtypes[k] ], yerr=np.array([regimeCorrArr_yerr[count].loc["DSW", windtypes[k] ]]).T, label=labels[3], color="b", fmt="o", capsize=3, markersize=3)
                axarr[-1].errorbar( (k+1)+2.*wth, regimeCorrArr[count].loc["ISW", windtypes[k] ], yerr=np.array([regimeCorrArr_yerr[count].loc["ISW", windtypes[k] ]]).T, label=labels[4], color="slategray", fmt="^", capsize=3, markersize=3)

            axarr[-1].tick_params(axis="x", direction="in")
            axarr[-1].axhline(y=0)
            axarr[-1].set_title(titles[count])
            axarr[-1].set_ylim(-1, 1)
            axarr[-1].set_xticklabels([None]+["-U$_{Slope}$", "V$_{Slope}$", "-$( \\vec{\\nabla} \\times \\vec{\\tau} )$", "W$_{shelf}$"], rotation=0)
            axarr[-1].set_yticks(np.arange(-1, 1.1, 0.25))
            if((i == mrows-1) ):
                axarr[-1].set_xlabel("Wind forcing")
            if((j == 0)):
                axarr[-1].set_ylabel("Correlation")
            else:
                axarr[-1].set_yticklabels([])
                axarr[-1].set_ylabel("")
            axarr[-1].grid(linestyle=":")    
            count+=1
    
    handles, labels = axarr[-1].get_legend_handles_labels()
    axlegend = plt.subplot(gs[mrows, :])
    axlegend.legend(handles, labels, ncol=3, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    if save:
        plt.savefig(savename, dpi=600)
    plt.show()




def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, round(p,4), lo, hi, [lo,hi]


def calculate_pvalues(df, alpha=0.05, retValue="yerr", maskInsignificant=False):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    rvalues = dfcols.transpose().join(dfcols, how='outer')
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    CI_lo = dfcols.transpose().join(dfcols, how='outer')
    CI_hi = dfcols.transpose().join(dfcols, how='outer')
    CI = dfcols.transpose().join(dfcols, how='outer')
    yerr = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            rvalues[r][c], pvalues[r][c], CI_lo[r][c], CI_hi[r][c], CI[r][c] = pearsonr_ci(df[r], df[c], alpha=alpha)
            yerr[r][c] = [(rvalues[r][c] - CI_lo[r][c]), (CI_hi[r][c] - rvalues[r][c]) ]
            if maskInsignificant:
                if((rvalues[r][c] < 0) & (CI[r][c][1] > -0.1) ):
                        rvalues[r][c] = np.nan
                        CI[r][c] = [np.nan, np.nan]
                if((rvalues[r][c] > 0) & (CI[r][c][0] < 0.1) ):
                        rvalues[r][c] = np.nan
                        CI[r][c] = [np.nan, np.nan]

    try:
        if retValue == "r":
            return rvalues
        if(retValue == "p"):
            return pvalues
        if(retValue == "lo"):
            return CI_lo
        if(retValue == "hi"):
            return CI_hi
        if(retValue == "CI"):
            return CI
        if(retValue == "yerr"):
            return yerr
    except:
        raise ValueError('retValue should be one of r,p,lo,hi')


def plot_WaterMass_Correlation_byRegion_CI(waterMassThickness, regionsName=[], titles=[], wd=7.48, ht=9, mrows=4, ncols=3, save=False, savename="untitled.png",
                                            windtypes=['neg_u10SlopeMonMean', 'v10MonMean', 'neg_stress_curl'], maskInsignificant=False):
    matplotlib.rcParams.update({'font.size': 8})        
    plt.close(1)
    fig = plt.figure(1, figsize=(wd,ht))    
        
    gs = gridspec.GridSpec(mrows+1, ncols, height_ratios=[1]*mrows+[0.1], width_ratios=[1]*ncols)
    gs.update(hspace=0.4)
    gs.update(wspace=0.1)

    wth = 0.1
    axarr = []
    count = 0
    for i in range(mrows):
        for j in range(ncols):
            axarr.append(plt.subplot(gs[i,j]))
            
            for k in range(len(windtypes)):
                if(k == 0):
                    labels = ["CDW", "mCDW", "LSSW", "DSW", "ISW"]
                else:
                    labels=[None]*5
                wmSelMask = waterMassThickness.region.str.contains(regionsName[count])
                wmCorr = calculate_pvalues(waterMassThickness[wmSelMask], retValue="r", maskInsignificant=maskInsignificant)
                wmCorr_yerr = calculate_pvalues(waterMassThickness[wmSelMask], retValue="yerr", maskInsignificant=maskInsignificant)

                axarr[-1].errorbar( (k+1)-2.*wth, wmCorr.loc["CDW", windtypes[k] ], yerr=np.array([wmCorr_yerr.loc["CDW", windtypes[k]]]).T, label=labels[0], color="r", fmt="x", capsize=3)
                axarr[-1].errorbar( (k+1)-1.*wth, wmCorr.loc["mCDW", windtypes[k] ], yerr=np.array([wmCorr_yerr.loc["mCDW", windtypes[k]]]).T, label=labels[1], color="darkorange", fmt="_", capsize=3)
                axarr[-1].errorbar( (k+1)+0.*wth, wmCorr.loc["LSSW", windtypes[k] ], yerr=np.array([wmCorr_yerr.loc["LSSW", windtypes[k]]]).T, label=labels[2], color="aqua", fmt="v", capsize=3)
                axarr[-1].errorbar( (k+1)+1.*wth, wmCorr.loc["DSW", windtypes[k] ], yerr=np.array([wmCorr_yerr.loc["DSW", windtypes[k]]]).T, label=labels[3], color="b", fmt="o", capsize=3)
                axarr[-1].errorbar( (k+1)+2.*wth, wmCorr.loc["ISW", windtypes[k] ], yerr=np.array([wmCorr_yerr.loc["ISW", windtypes[k]]]).T, label=labels[4], color="slategray", fmt="^", capsize=3)
            axarr[-1].axhline(y=0)
            axarr[-1].set_title(titles[count])
            axarr[-1].set_xticks([1,1.5, 2,2.5, 3])
            if(i == mrows-1):
                axarr[-1].set_xticklabels(["-U$_{Slope}$", "", "V", "", "$-(\\nabla \\times \\tau)_{slope}$"], rotation=0)
            else:
                axarr[-1].set_xticklabels([])
                axarr[-1].set_xlabel("")
            
            if((i == mrows-1) & (j == 1) ):
                axarr[-1].set_xlabel("Wind types")
            if((j == 0)):
                axarr[-1].set_ylabel("Correlation")
            else:
                axarr[-1].set_yticklabels([])
                axarr[-1].set_ylabel("")
            axarr[-1].set_ylim(-1, 1)
            axarr[-1].set_yticks(np.arange(-1,1.1,0.25))
            axarr[-1].grid(linestyle=":")
            count+=1
    
    handles, labels = axarr[-1].get_legend_handles_labels()
    axlegend = plt.subplot(gs[mrows, :])
    axlegend.legend(handles, labels, ncol=5, loc=9)
    axlegend.set_yticklabels([])
    axlegend.set_yticks([])
    axlegend.set_xticklabels([])
    axlegend.set_xticks([])
    axlegend.spines["top"].set_visible(False)
    axlegend.spines["right"].set_visible(False)
    axlegend.spines["left"].set_visible(False)
    axlegend.spines["bottom"].set_visible(False)

    if save:
        plt.savefig(savename, dpi=600)
    plt.show()
