import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors
from haversine import haversine
from matplotlib.colors import LinearSegmentedColormap

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
    
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='h');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    dist_x = 8199.491128244028
    w_factor = 0.3
    h_factor = 0.3
    if(region =='Whole'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*w_factor, llcrnry=-height*h_factor, urcrnrx=w_factor*width, urcrnry=h_factor*height)
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*0.225, llcrnry=-0.1*height, urcrnrx=0.*width, urcrnry=0.20*height)

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=0.2)    

    parallels = np.arange(-80, -50+1, 5.)    
    m.drawparallels(parallels,labels=[1,1,1,0], linewidth=0.2)
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1], linewidth=0.2)

    return m

def getCellSize(m, nx=300, ny=300):
    width = m.urcrnrx - m.llcrnrx
    height = m.urcrnry - m.llcrnry
    left = [m.llcrnrx, m.llcrnry+height*0.5]
    right = [m.urcrnrx, m.urcrnry-height*0.5]
    bottom = [m.llcrnrx+width*0.5, m.llcrnry]
    top = [m.llcrnrx+width*0.5, m.urcrnry]
    
    left_lonlat = m(left[0], left[1], inverse=True)
    right_lonlat = m(right[0], right[1], inverse=True)
    top_lonlat = m(top[0], top[1], inverse=True)
    bot_lonlat = m(bottom[0], bottom[1], inverse=True)
    dist_x = haversine(list(left_lonlat[::-1]), list(right_lonlat[::-1]))
    dist_y = haversine(top_lonlat[::-1], bot_lonlat[::-1])
    print("Horizontal distance = ", dist_x, "\n", "Vertical distance = ", dist_y)
    
def plotBotVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png", wd=7, ht=7, cmap='viridis', region='Whole', nmin=0, show=False, nx=820, ny=820):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m = createMapProjections(lat0, lon0, region=region)

    botIndex = df.groupby('PROFILE_NUMBER').tail(1).index
    
    z = df.loc[botIndex, var].values
    xlons, ylats = m(df.loc[botIndex , 'LONGITUDE'].values, df.loc[botIndex,'LATITUDE'].values)

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
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    m.drawcoastlines(linewidth=0.2)

    ticks = np.arange(cmin, cmax, 0.2)
    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, boundaries=ticks, spacing='uniform') 
    cbar.set_label(units)

    if(save==True):
        plt.savefig(savename, dpi=300)

    if(show == True):
        plt.show();
    else:
        plt.close();


def plotSurfVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png",
                         wd=7, ht=7, nx=820, ny=820, cmap='viridis', nmin=30, region='Whole', levels=[0], show=False):
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

    if(len(levels) == 1):
        ticks = np.arange(cmin, cmax, 0.2)
    else:
        ticks = levels

    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad='12%', boundaries=ticks, ticks=ticks, spacing='uniform') 
    cbar.set_label(units)
    
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();


def plotDataDensity(df, units='Data Density',
                    save=False, savename="savedFig.png", wd=7, ht=7,
                    nx=820, ny=820, show=False,
                    levels=[0, 10, 20, 30, 40, 50, 60]):  #, 90, 150, 250, 500, 1000, 2500]):
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
    ticks = levels
    from matplotlib.colors import BoundaryNorm
    bnorm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)

    cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=['mistyrose', 'salmon', 'darkred'],
                                             N=len(levels)-1)
    CF = m.contourf(XX, YY, ni_masked, vmin=min(levels), vmax=max(levels), levels=levels, norm=bnorm,
                    cmap=cmap, extend='max'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, spacing='uniform') #boundaries=levels, 
    
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();


def plotDepthOfDive(df, units='Data Density',
                    save=False, savename="savedFig.png", wd=7, ht=7,
                    nx=820, ny=820, deltac=50, show=False,
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

    levels= levels[::-1]
    ticks = levels
    from matplotlib.colors import BoundaryNorm
    bnorm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)

    cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=['darkblue', 'CornFlowerBlue', 'w'], N=len(levels)-1)
    CF = m.contourf(XX, YY, zi_masked, vmin=min(levels), vmax=0, levels=levels, norm=bnorm,
                    cmap=cmap, extend='max'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad=0.6, ticks=ticks, spacing='uniform') #boundaries=levels, 
    
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();


def plotMaxVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png", show=False,
                         wd=7, ht=7, nx=820, ny=820, cmap='viridis', nmin=30, region='Whole', levels=[0], isvar=True):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m  = createMapProjections(lat0, lon0, region=region)
    
    surfIndex = df.groupby('PROFILE_NUMBER')[var].transform(max) == df[var]
    
    if(isvar == False):
        var = 'DEPTH'
        levels= np.array([0, -100, -200, -300, -400, -500, -600, -700, -800])[::-1]
    
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

    
    if(len(levels) == 1):
        ticks = np.arange(cmin, cmax, 0.2)
    else:
        ticks = levels
        cmin = min(levels)
        cmax = max(levels)

    CF = m.contourf(XX, YY, zi_masked, vmin=cmin, vmax=cmax, cmap=cmap, levels=ticks, extend='both'); #, cmap='viridis'
    cbar = m.colorbar(CF, pad='12%', boundaries=ticks, ticks=ticks, spacing='uniform') 
    cbar.set_label(units)
    
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();