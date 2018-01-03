import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors


def createMapProjections(lat0, lon0, region='Whole'):
    #m = Basemap(projection='hammer',lon_0=270)
    # plot just upper right quadrant (corners determined from global map).
    # keywords llcrnrx,llcrnry,urcrnrx,urcrnry used to define the lower
    # left and upper right corners in map projection coordinates.
    # llcrnrlat,llcrnrlon,urcrnrlon,urcrnrlat could be used to define
    # lat/lon values of corners - but this won't work in cases such as this
    # where one of the corners does not lie on the earth.
    #m1 = Basemap(projection='ortho',lon_0=-9,lat_0=60,resolution=None)
    
    #m = Basemap(projection='cyl', llcrnrlon=-120, llcrnrlat=-80, urcrnrlon=55, 
    #            urcrnrlat=-30, lat_0 = -74, lon_0 =-43);
    
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='i');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    if(region =='Whole'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='i', llcrnrx=-width*0.30, llcrnry=-0.30*height, urcrnrx=0.30*width, urcrnry=0.30*height)
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='i', llcrnrx=-width*0.225, llcrnry=-0.1*height, urcrnrx=0.*width, urcrnry=0.20*height)
    return m

def plotBotVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png", wd=7, ht=7, cmap='viridis', region='Whole', levels=[0], nmin=0):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m = createMapProjections(lat0, lon0, region=region)

    botIndex = df.groupby('PROFILE_NUMBER').tail(1).index
    
    z = df.loc[botIndex, var].values
    xlons, ylats = m(df.loc[botIndex , 'LONGITUDE'].values, df.loc[botIndex,'LATITUDE'].values)

    nx, ny = 300, 300
    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    zi = np.zeros((len(Ygrid), len(Xgrid)))
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(z)):
        zi[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += z[i]
        ni[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += 1

    wz = np.where(zi == 0) # where zero
    ni[wz[0], wz[1]] = 1
    zi[:, :] = zi[:, :]/ni[:, :]
    zi = ma.array(zi)
    zi_masked = ma.masked_where(zi == 0, zi)
    ni_masked = ma.masked_less(ni, nmin)
    zi_masked = ma.masked_array(data=zi, mask=(zi_masked.mask|ni_masked.mask))

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75')
    m.drawcoastlines(linewidth=0.5)

    if(len(levels) == 1):
        ticks = np.arange(cmin, cmax, 0.2)
    else:
        ticks = levels

    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, boundaries=ticks, spacing='uniform') 
    cbar.set_label(units)
    parallels = np.arange(-80, -50+1, 5.)
    
    m.drawparallels(parallels,labels=[1,1,0,0])
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,1,1])

    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();


def plotSurfVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png",
                         wd=7, ht=7, nx=300, ny=300, cmap='viridis', nmin=30, region='Whole', levels=[0]):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m  = createMapProjections(lat0, lon0, region=region)
    
    #lat_0 = -60, lon_0 = -20,

    surfIndex = df.groupby('PROFILE_NUMBER').head(1).index
    
    z = df.loc[surfIndex, var].values
    xlons, ylats = m(df.loc[surfIndex , 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    zi = np.zeros((len(Ygrid), len(Xgrid)))
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(z)):
        zi[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += z[i]
        ni[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += 1

    wz = np.where(zi == 0) # where zero
    ni[wz[0], wz[1]] = 1 #setting 0 counts to 1 to avoid div by zero
    zi[:, :] = zi[:, :]/ni[:, :] #averaging over the no of samples from each grid cell
    zi = ma.array(zi)
    zi_masked = ma.masked_where(zi == 0, zi)
    ni_masked = ma.masked_less(ni, nmin)
    zi_masked = ma.masked_array(data=zi, mask=(zi_masked.mask|ni_masked.mask))

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75')
    m.drawcoastlines(linewidth=0.5)
    
    if(len(levels) == 1):
        ticks = np.arange(cmin, cmax, 0.2)
    else:
        ticks = levels

    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad='12%', boundaries=ticks, ticks=ticks, spacing='uniform') 
    cbar.set_label(units)
    
    parallels = np.arange(-80, -50+1, 5.)
    
    m.drawparallels(parallels,labels=[1,1,1,0])
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1])

    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();


def plotDataDensity(df, units='Data Density',
                         save=False, savename="savedFig.png", wd=7, ht=7,
                          nx=300, ny=300,
                          levels=[0, 30, 60, 90, 150, 250, 500, 1000, 2500]):
    plt.figure(figsize=(wd,ht));
    lat0 = -89
    lon0 = 0
            
    m  = createMapProjections(lat0, lon0)
    
    #lat_0 = -60, lon_0 = -20,

    surfIndex = df.groupby('PROFILE_NUMBER').head(1).index
    
    xlons, ylats = m(df.loc[surfIndex , 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(xlons)):
        row = np.argmin(np.abs(Ygrid - ylats[i]))
        col = np.argmin(np.abs(Xgrid - xlons[i]))
        ni[row , col] = ni[row, col] + 1

    ni = ma.array(ni)
    ni_masked = ma.masked_equal(ni ,0)

    #zi_masked = ma.masked_array(zi, 0)
    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75')
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=0.5)    
    ticks = levels
    from matplotlib.colors import BoundaryNorm
    bnorm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)

    cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=['mistyrose', 'salmon', 'darkred'],
                                             N=len(levels)-1)
    CF = m.contourf(XX, YY, ni_masked, vmin=min(levels), vmax=max(levels), levels=levels, norm=bnorm,
                    cmap=cmap, extend='max'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, spacing='uniform') #boundaries=levels, 
    
    parallels = np.arange(-80, -50+1, 5.)
    
    m.drawparallels(parallels,labels=[1,1,1,0])
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1])

    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();
    #for i in range(len(xgrid)):
        #plt.annotate(str(stations[i]), (xgrid[i],ygrid[i]))


def plotDepthOfDive(df, units='Data Density',
                    save=False, savename="savedFig.png", wd=7, ht=7,
                    nx=300, ny=300, deltac=50,
                    levels=[0, -50, -100, -200, -300, -400, -500, -750, -1000, -1500, -2000], nmin=30):
    
    plt.figure(figsize=(wd,ht));
    lat0 = -89
    lon0 = 0
            
    m  = createMapProjections(lat0, lon0)
    
    #lat_0 = -60, lon_0 = -20,

    botIndex = df.groupby('PROFILE_NUMBER').tail(1).index
    
    xlons, ylats = m(df.loc[botIndex , 'LONGITUDE'].values, df.loc[botIndex,'LATITUDE'].values)
    z = df.loc[botIndex, 'DEPTH'].values
    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    zi = np.zeros((len(Ygrid), len(Xgrid)))
    ni = np.zeros((len(Ygrid), len(Xgrid)))
        
    for i in range(len(xlons)):
        row = np.argmin(np.abs(Ygrid - ylats[i]))
        col = np.argmin(np.abs(Xgrid - xlons[i]))
        zi[row , col] = z[i]
        ni[row, col] += 1

    zi = ma.array(zi)
    zi_masked = ma.masked_equal(zi , 0)
    ni_masked = ma.masked_less(ni, nmin)
    zi_masked = ma.masked_array(data=zi, mask=(zi_masked.mask|ni_masked.mask))

    #zi_masked = ma.masked_array(zi, 0)
    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75')
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=0.5)
    levels= levels[::-1]
    ticks = levels
    from matplotlib.colors import BoundaryNorm
    bnorm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)

    cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=['darkblue', 'CornFlowerBlue', 'w'], N=len(levels)-1)
    CF = m.contourf(XX, YY, zi_masked, vmin=min(levels), vmax=0, levels=levels, norm=bnorm,
                    cmap=cmap, extend='max'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, spacing='uniform') #boundaries=levels, 
    
    parallels = np.arange(-80, -50+1, 5.)
    
    m.drawparallels(parallels,labels=[1,1,1,0])
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1])

    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();


def plotMaxVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png",
                         wd=7, ht=7, nx=300, ny=300, cmap='viridis', nmin=30, region='Whole', levels=[0]):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m  = createMapProjections(lat0, lon0, region=region)
    
    #lat_0 = -60, lon_0 = -20,

    surfIndex = df.groupby('PROFILE_NUMBER')[var].transform(max) == df[var]
    
    z = df.loc[surfIndex, var].values
    xlons, ylats = m(df.loc[surfIndex , 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    zi = np.zeros((len(Ygrid), len(Xgrid)))
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(z)):
        zi[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += z[i]
        ni[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += 1

    wz = np.where(zi == 0) # where zero
    ni[wz[0], wz[1]] = 1 #setting 0 counts to 1 to avoid div by zero
    zi[:, :] = zi[:, :]/ni[:, :] #averaging over the no of samples from each grid cell
    zi = ma.array(zi)
    zi_masked = ma.masked_where(zi == 0, zi)
    ni_masked = ma.masked_less(ni, nmin)
    zi_masked = ma.masked_array(data=zi, mask=(zi_masked.mask|ni_masked.mask))

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75')
    m.drawcoastlines(linewidth=0.5)
    
    if(len(levels) == 1):
        ticks = np.arange(cmin, cmax, 0.2)
    else:
        ticks = levels

    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad='12%', boundaries=ticks, ticks=ticks, spacing='uniform') 
    cbar.set_label(units)
    
    parallels = np.arange(-80, -50+1, 5.)
    
    m.drawparallels(parallels,labels=[1,1,1,0])
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1])

    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();
