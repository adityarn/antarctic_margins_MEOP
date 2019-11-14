import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors
from haversine import haversine
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import matplotlib
import xarray as xr
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar
import mplcursors
import cartopy.crs as ccrs
import cartopy
import fiona
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def createMapProjections(lat0, lon0, region='Whole', fontsize=8, annotate=True, ax=None, linewidth=0.2, draw_grounding_line=True, regionLonLims=None):
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
    matplotlib.rcParams.update({'font.size': fontsize})    
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='l');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    dist_x = 8199.491128244028
    w_factor = 0.3
    h_factor = 0.3
    if(region =='Whole'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*w_factor, llcrnry=-height*h_factor, urcrnrx=w_factor*width, urcrnry=h_factor*height)
        ## regionNames = ['Weddell Sea (WS)', 'Ross Sea (RS)', 'Prydz Bay (PB)', 'Cape Darnley (CD)', 'Knox Coast (KC)', 'Adelie Coast (AC)', 'Amundsen Sea (AS)', 'Belingshausen Sea (BS)']
        ## regionCoord = [[-45, -65, 40], [180 , -70, 0], [75, -50, -30], [65, -50, -25], [100, -55, 10], [140, -55, 40], [-100, -60, -10], [-90, -60, 90]]
        ## for i in range(len(regionNames)):
        ##     mxy = m(regionCoord[i][0], regionCoord[i][1] )
        ##     plt.annotate(regionNames[i], xy=(mxy[0], mxy[1]), rotation=regionCoord[i][2], fontsize=12)
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*0.15, llcrnry=0.05*height, urcrnrx=0.*width, urcrnry=0.18*height)
    elif(region == 'Ross'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=180, resolution='h', llcrnrx=-width*0.075, llcrnry=height*0.08, urcrnrx=width*0.03, urcrnry=height*0.18)
    elif(region == 'CDP'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=65, resolution='h', llcrnrx=-width*0.05, llcrnry=height*0.175, urcrnrx=width*0.05, urcrnry=height*0.23)
    elif(region == 'Prydz'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=75, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.15, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Knox'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=106.5, resolution='h', llcrnrx=-width*0.07, llcrnry=height*0.17, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Adelie'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=140, resolution='h', llcrnrx=-width*0.07, llcrnry=height*0.17, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Amundsen'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=-110, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.12, urcrnrx=width*0.07, urcrnry=height*0.18)
    elif(region == 'Belingshausen'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=-75, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.12, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Global'):
        m = m1

    #m.drawmapboundary(linewidth=linewidth);
    if(draw_grounding_line == True):
        m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=linewidth)    

    if regionLonLims:
        ## Draw the regions longitudinal limits
        m.drawmeridians(regionLonLims, labels=[0,0,0,0], linewidth=0.5, color='b')    # labels: left,right,top,bottom

    if(annotate == True):
        parallels = np.arange(-80, -50+1, 5.)    
        m.drawparallels(parallels,labels=[0,0,0,0], linewidth=0.2) # labels: left,right,top,bottom
        meridians = np.arange(-180, 180, 20.)
        m.drawmeridians(meridians,labels=[1,1,1,1], linewidth=0.2)
        
        xy = [[-140, -55], [-140, -60] , [-140, -65], [-140, -70], [-140, -75] , [-140, -80]]
        xytext = np.arange(55, 81, 5)
        for i in range(len(xytext)):
            mxy = m(xy[i][0], xy[i][1])
            plt.annotate(str(xytext[i])+"$^o$S", xy=(mxy[0], mxy[1]), rotation=-45, fontsize=fontsize)
    return m


def createMapProjections_CRS(lat0, lon0, region='Whole', fontsize=8, annotate=True, ax=None, linewidth=0.2, draw_grounding_line=True, regionLonLims=None):
    matplotlib.rcParams.update({'font.size': fontsize})

    if(region =='Whole'):
        m = ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0)
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*0.15, llcrnry=0.05*height, urcrnrx=0.*width, urcrnry=0.18*height)
    elif(region == 'Ross'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=180, resolution='h', llcrnrx=-width*0.075, llcrnry=height*0.08, urcrnrx=width*0.03, urcrnry=height*0.18)
    elif(region == 'CDP'):
        m = ccrs.PlateCarree()
    elif(region == 'Prydz'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=75, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.15, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Knox'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=106.5, resolution='h', llcrnrx=-width*0.07, llcrnry=height*0.17, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Adelie'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=140, resolution='h', llcrnrx=-width*0.07, llcrnry=height*0.17, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Amundsen'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=-110, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.12, urcrnrx=width*0.07, urcrnry=height*0.18)
    elif(region == 'Belingshausen'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=-75, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.12, urcrnrx=width*0.07, urcrnry=height*0.25)
    elif(region == 'Global'):
        m = m1

    #m.drawmapboundary(linewidth=linewidth);
    if(draw_grounding_line == True):
        m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=linewidth)    

    if regionLonLims:
        ## Draw the regions longitudinal limits
        m.drawmeridians(regionLonLims, labels=[0,0,0,0], linewidth=0.5, color='b')    # labels: left,right,top,bottom

    if(annotate == True):
        parallels = np.arange(-80, -50+1, 5.)    
        m.drawparallels(parallels,labels=[0,0,0,0], linewidth=0.2) # labels: left,right,top,bottom
        meridians = np.arange(-180, 180, 20.)
        m.drawmeridians(meridians,labels=[1,1,1,1], linewidth=0.2)
        
        xy = [[-140, -55], [-140, -60] , [-140, -65], [-140, -70], [-140, -75] , [-140, -80]]
        xytext = np.arange(55, 81, 5)
        for i in range(len(xytext)):
            mxy = m(xy[i][0], xy[i][1])
            plt.annotate(str(xytext[i])+"$^o$S", xy=(mxy[0], mxy[1]), rotation=-45, fontsize=fontsize)
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

    return dist_x/float(nx), dist_y/float(ny)
    
def plotBotVarContourf(df,var="PSAL_ADJUSTED", units='PSU', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png", wd=7, ht=7, cmap='viridis', region='Whole', nmin=0, show=True, cx=10, cy=10, levs=[]):

    matplotlib.rcParams.update({'font.size': 8})        # setting fontsize for plot elements            
    plt.figure(1, figsize=(wd,ht));
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05])
    mapax = plt.subplot(gs[0, 0])

    lat0 = -90
    lon0 = 0
    m = createMapProjections(lat0, lon0, region=region)
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
    nx = dist_x // cx
    ny = dist_x // cy
    print(nx, ny)

    botIndex = df.groupby('PROFILE_NUMBER').tail(1).index
    
    z = df.loc[botIndex, var].values
    xlons, ylats = m(df.loc[botIndex , 'LONGITUDE'].values, df.loc[botIndex,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    zi = np.zeros((len(Ygrid), len(Xgrid)))
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(z)):
        if(np.isnan(z[i]) == True):
            pass
        else:
            zi[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += z[i]
            ni[np.argmin(np.abs(Ygrid - ylats[i])), np.argmin(np.abs(Xgrid - xlons[i]))] += 1

    #wz = np.where(zi == 0) # where zero
    #ni[wz[0], wz[1]] = 1
    zi[:, :] = zi[:, :]/ni[:, :]
    #zi = ma.array(zi)
    #zi_masked = ma.masked_where(zi == 0, zi)
    #ni_masked = ma.masked_less(ni, nmin)
    #zi_masked = ma.masked_array(data=zi, mask=(zi_masked.mask|ni_masked.mask))
    
    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    m.drawcoastlines(linewidth=0.2)

    if not levs:
        levs = np.arange(cmin, cmax, 0.2)
    CF = m.contourf(XX, YY, zi, vmin=cmin, vmax=cmax, cmap=cmap, levels=levs, extend='max', ax=mapax)
    colorbarax = plt.subplot(gs[0,1])
    cbar = Colorbar(ax=colorbarax, mappable=CF, ticks=levs) 
    cbar.set_label(units)

    if(save==True):
        plt.savefig(savename, dpi=300, bbox_inches='tight')

    if(show == True):
        plt.show();
    else:
        plt.close();


def plotSurfVarContourf(df,var="PSAL_ADJUSTED", units='Cond.', cmin=33, cmax=35.5,
                         save=False, savename="savedFig.png",
                         wd=7, ht=7, cx=10, cy=10, cmap='viridis', nmin=0, region='Whole', levels=[0], show=True):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m  = createMapProjections(lat0, lon0, region=region)
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
    nx = dist_x // cx
    ny = dist_x // cy
    print(nx, ny)
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
                    cx=10, cy=10, show=False, lat0 = -90, lon0=0, regionLonLims = [ -60. ,  -20. ,  0., 29., 37., 60. ,   70. , 75,  82. ,  87., 101. ,  112. ,
        135. ,  145. ,  160. ,  180. , -120. , -100.],
                    levels=[0, 10, 20, 30, 40, 50, 60, 100, 200, 500], region='Whole', plotBathy=False, fontsize=8,
                    bathyLevels=[0, -100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1250, -1500, -1750, -2000, -2500, -3000, -3500, -4000, -4500, -5000, -6000, -7000]):
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements            
    plt.figure(1, figsize=(wd,ht));
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.15], width_ratios=[1, 0.03])
    mapax = plt.subplot(gs[0:2, 0])
    m  = createMapProjections(lat0, lon0, region=region, fontsize=fontsize, regionLonLims= regionLonLims)

        
    datacolorbar = plt.subplot(gs[0, 1])
    #bathycolorbar = plt.subplot(gs[1, 1])
    
    matplotlib.rcParams.update({'font.size': fontsize})    
    

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
    nx = dist_x // cx
    ny = dist_x // cy
    #lat_0 = -60, lon_0 = -20,

    surfIndex = df.groupby('PROFILE_NUMBER').head(1).index
    
    xlons, ylats = m(df.loc[surfIndex , 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    ni = np.zeros((len(Ygrid), len(Xgrid)))
    
    if(plotBathy == True):
        bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-1000, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = m.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  latlon=True, levels=[-1000], colors="magenta", linestyle=":", linewidths=0.35, extend='neither', ax=mapax) #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        
        #cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'vertical')
        #cbar1.ax.get_children()[0].set_linewidths(5)
        #cbar1.set_label('Depth (m)')
        
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

    cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=['mistyrose', 'salmon', 'darkred'], N=len(levels)-1) 
        
    CF = m.contourf(XX, YY, ni_masked, vmin=min(levels), vmax=max(levels), levels=levels, norm=bnorm,
                    cmap=cmap, extend='max', ax=mapax); #, cmap='viridis'
    cbar = Colorbar(ax = datacolorbar, mappable = CF, ticks=ticks, orientation='vertical') #boundaries=levels, 
    cbar.set_label("Data density")


    if(save==True):
        plt.savefig(savename, dpi=300, bbox_inches='tight')
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
                         save=False, savename="savedFig.png", show=True,
                         wd=7, ht=7, cx=10, cy=10, cmap='viridis', nmin=30, region='Whole', levels=[0], isvar=True):
    plt.figure(1,figsize=(wd,ht));
            
    lat0 = -89
    lon0 = 0
    m  = createMapProjections(lat0, lon0, region=region)
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
    nx = dist_x // cx
    ny = dist_x // cy
    print(nx, ny)
    
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


def plotSeaIceConc(ICE, timeind, latmin=-90, latmax=-60, lonmin=-180, lonmax=180, save=False, savename="savedFig.png", show=False,
                         wd=7, ht=7, nx=820, ny=820, cmap='viridis', nmin=30, region='Whole', levels=np.arange(0,1.1,0.1), isvar=True):

    sic = ICE.sic
    
    fig = plt.figure(figsize=(wd,ht));
            
    ax = plt.axes(projection=ccrs.Orthographic(0, -90))
    #cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])

    sic_contour = sic.isel(time=timeind).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=np.arange(0,1.1,0.1));
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    def resize_colobar(event):
        plt.draw()

        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                              0.04, posn.height])

    #fig.canvas.mpl_connect('resize_event', resize_colobar)
    #plt.colorbar(sic_contour, cax=cbar_ax)
    #ax.set_global();
    ax.coastlines();

        

    if(show==True):
        plt.show();
    if(save == True):
        plt.savefig(savename)
    

def find_area(df, m, units='Data Density',
                    nx=820, ny=820, lat0 = -90, lon0=0,region='Whole', years=[]):
    if not years:
        years = np.sort(df.JULD.dt.year.unique())
    area = np.zeros(len(years))
    
    for j in range(len(years)):
        year_mask = df['JULD'].dt.year == years[j]
        surfIndex = df[year_mask].groupby('PROFILE_NUMBER').head(1).index

        xlons, ylats = m(df[year_mask].loc[surfIndex , 'LONGITUDE'].values, df[year_mask].loc[surfIndex,'LATITUDE'].values)

        Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
        Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
        XX, YY = np.meshgrid(Xgrid, Ygrid)
        ni = np.zeros((len(Ygrid), len(Xgrid)))

        for i in range(len(xlons)):
            row = np.argmin(np.abs(Ygrid - ylats[i]))
            col = np.argmin(np.abs(Xgrid - xlons[i]))
            ni[row , col] = 1

        ni = ma.array(ni)
        ni_masked = ma.masked_equal(ni ,0)

        cell_size_x, cell_size_y = getCellSize(m, nx=nx, ny=ny)

        number_of_cells = np.sum(ni_masked)

        area[j] = cell_size_x * cell_size_y * number_of_cells
    
    return area

def plotDataDensity2layered(ax, df,mask1,mask2, units='Data Density',
                    save=False, savename="savedFig.png", wd=7, ht=7,
                    cx=10, cy=10, show=False, lat0 = -90, lon0=0, annotate=True,
                    levels=[1,5,10,20,30,50], region='Whole', m=None, fontsize=8):  #, 90, 150, 250, 500, 1000, 2500]):
                        
    matplotlib.rcParams.update({'font.size': fontsize})        
    #fig = plt.figure(figsize=(wd,ht));
    if(m == None):
        m  = createMapProjections(lat0, lon0, region=region, annotate=annotate)

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
    nx = dist_x // cx
    ny = dist_x // cy
    #lat_0 = -60, lon_0 = -20,

    surfIndex1 = df[mask1].groupby('PROFILE_NUMBER').head(1).index
    xlons1, ylats1 = m(df[mask1].loc[surfIndex1 , 'LONGITUDE'].values, df[mask1].loc[surfIndex1,'LATITUDE'].values)
    surfIndex2 = df[mask2].groupby('PROFILE_NUMBER').head(1).index
    xlons2, ylats2 = m(df[mask2].loc[surfIndex2 , 'LONGITUDE'].values, df[mask2].loc[surfIndex2,'LATITUDE'].values)

    Xgrid = np.linspace(m.llcrnrx, m.urcrnrx, nx)
    Ygrid = np.linspace(m.llcrnry, m.urcrnry, ny)
    XX, YY = np.meshgrid(Xgrid, Ygrid)
    ni1 = np.zeros((len(Ygrid), len(Xgrid)))
    ni2 = np.zeros((len(Ygrid), len(Xgrid)))
    
    for i in range(len(xlons1)):
        row = np.argmin(np.abs(Ygrid - ylats1[i]))
        col = np.argmin(np.abs(Xgrid - xlons1[i]))
        ni1[row , col] = ni1[row, col] + 1
        
    for i in range(len(xlons2)):
        row = np.argmin(np.abs(Ygrid - ylats2[i]))
        col = np.argmin(np.abs(Xgrid - xlons2[i]))
        ni2[row , col] = ni2[row, col] + 1

    ni1 = ma.array(ni1)
    ni1_masked = ma.masked_equal(ni1 ,0)
    ni2 = ma.array(ni2)
    ni2_masked = ma.masked_equal(ni2 ,0)

    #zi_masked = ma.masked_array(zi, 0)
    ticks = levels
    from matplotlib.colors import BoundaryNorm
    bnorm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=False);
    
    cmap1 = LinearSegmentedColormap.from_list(name='linearCmap', colors=['mistyrose', 'salmon', 'darkred'], N=len(levels)-1); 
    cmap2 = LinearSegmentedColormap.from_list(name='linearCmap', colors=['paleturquoise', 'deepskyblue', 'navy'], N=len(levels)-1);
        
    CF1 = m.contourf(XX, YY, ni1_masked, vmin=min(levels), vmax=max(levels), levels=levels, norm=bnorm,
                    cmap=cmap1, extend='max', ax=ax); #, cmap='viridis'
    CF2 = m.contourf(XX, YY, ni2_masked, vmin=min(levels), vmax=max(levels), levels=levels, norm=bnorm,
                    cmap=cmap2, extend='max', ax=ax); #, cmap='viridis'

    ## cbar2 = m.colorbar(CF2, ticks=ticks, spacing='uniform', pad=0.6, location='bottom') #cax=cbaxes2) #boundaries=levels,
    ## cbar2.set_label('Area II, beyond 75km from GL')
    ## cbar1 = m.colorbar(CF1,  ticks=ticks, spacing='uniform', pad=0.6) #cax=cbaxes1) #boundaries=levels,    
    ## cbar1.set_label('Area I, within 75km of GL')

    ## plt.tight_layout();
    ## if(save==True):
    ##     plt.savefig(savename, dpi=300, bbox_inches='tight')
    ## if(show == True):
    ##     plt.show();
    ## else:
    ##     plt.close();
    return CF1, CF2

def createShadedRegionPlot(regionLonLims=[], regionNames=[], regionNamesxy=[], wd=7, ht=7, lat0=-90, lon0=0, region="Whole",
                           save=False, savename="untitled.png", fontsize=8, annotate=True, linewidth=0.2, dpi=300):
    matplotlib.rcParams.update({'font.size': fontsize, 'figure.autolayout': True})
    if not regionLonLims:
        print("Error! No regions mask provided")
        return 0
    fig = plt.figure(figsize=(wd,ht));
    #fig.set_facecolor('0.75')
    m  = createMapProjections(lat0, lon0, region=region, fontsize=fontsize, annotate=annotate, linewidth=0.2, draw_grounding_line=False)
    m.drawmapboundary(fill_color="grey", zorder=1, linewidth=0.1)
    m.drawcoastlines(linewidth=0.1, zorder=3)

    ## Fill the regions with white color, but only where depth is below 0
    for i in range(len(regionLonLims)):
        lons = np.linspace(regionLonLims[i][0], regionLonLims[i][1], 10)
        lats = np.linspace(-80, -60, 10)
        x,y = m(lons, lats)
        longrid, latgrid = np.meshgrid(lons, lats)
        white = np.zeros((len(lats), len(lons))) #np.random.randn(len(lats), len(lons) )
        m.contourf(longrid, latgrid, white, vmin=0, vmax=1, cmap="Greys", latlon=True, zorder=2) # cmap="Greys"
        
    m.fillcontinents(color='grey', zorder=2);
    
    for i in range(len(regionNames)):
        mxy = m(regionNamesxy[i][0], regionNamesxy[i][1])
        plt.annotate(regionNames[i], xy=(mxy[0], mxy[1]), fontsize=fontsize)
    plt.tight_layout()
    if(save == True):
        plt.savefig(savename, dpi=dpi, bbox_inches="tight")
    
    plt.show()


def plot_station_locations(positions, title=' ', save=False, savename="untitled.png", wd=12, ht=12, region='Whole', plotBathy=True, m=None):
    x = positions[:,1]
    y = positions[:,0]
    
    lat0 = -90
    lon0 = 0
    
    plt.figure(1, figsize=(wd,ht));
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.025])
    mapax = plt.subplot(gs[0, 0])
    if not m:
        m  = createMapProjections(lat0, lon0, region=region)

    m.scatter(x, y, latlon=True, color='r', s=1, marker='.')
    m.drawcoastlines()
    if(plotBathy == True):
        bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-100, -500, -1000, -1500, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = m.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  latlon=True, levels=clevs, linewidths=0.2, extend='min', ax=mapax) #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        bathycolorbar = plt.subplot(gs[1, 0])
        cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'horizontal')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')

    if(save == True):
        plt.savefig(savename, dpi=300)#, bbox_inches='tight')
    plt.show()


def plotProfileNumberContours(df,var="PROFILE_NUMBER", plotBathy=True,
                         save=False, savename="savedFig.png",
                         wd=7, ht=7, cx=10, cy=10, cmap='viridis', nmin=0, region='Whole', levels=[0], show=True):
    plt.close(1)
    plt.figure(1,figsize=(wd,ht));
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    mapax = plt.subplot(gs[0,0])
        
    lat0 = -90
    lon0 = 0

    m  = createMapProjections(lat0, lon0, region=region)
    bathycolorbar = plt.subplot(gs[1, 0])
        
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
    nx = dist_x // cx
    ny = dist_x // cy
    print(nx, ny)
    surfIndex = df.groupby('PROFILE_NUMBER').head(1).index
    
    z = df.loc[surfIndex, var].values
    xlons, ylats = m(df.loc[surfIndex , 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values)

    if(plotBathy == True):
        bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-100, -500, -1000, -1500, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = m.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  latlon=True, levels=clevs, linewidths=0.2, extend='min', ax=mapax) #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'horizontal')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')

    
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();


def plot_scalar_field(sclr, lons, lats, m=None, bathy=None, plotBathy=True, save=False, savename="untitled.png", levs=None, drawMeridians=True, meridians=None, pickPoints=True):
    plt.close(1)
    fig = plt.figure(1, figsize=(10,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 0.05])
    mapax = plt.subplot(gs[0,0])
    colorbar_ax = plt.subplot(gs[1,0])
    bathycolorbar = plt.subplot(gs[0, 1])
    
    X, Y = np.meshgrid(lons, lats)
    
    if not m:
        m = createMapProjections(-90, 0, region="Whole")
        
    CF = m.contourf(X, Y, sclr, ax=mapax, latlon=True, levels=levs, extend="both")
    cbar1 = Colorbar(ax = colorbar_ax, mappable = CF, orientation = 'horizontal')
    
    if(plotBathy == True):
        if not bathy:
            bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-100, -500, -1000, -1500, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = m.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  latlon=True, 
                       levels=clevs, linewidths=0.2, extend='min', ax=mapax, colors='k' ) #, cmap="RdYlBu"
        cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'vertical')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')
        
    if drawMeridians:
        if not meridians:
            meridians = np.arange(-180, 180, 20)
        m.drawmeridians(meridians,labels=[1,1,1,1], linewidth=0.2, ax=mapax)
    if save:
        plt.savefig(savename, dpi=600)
    m.drawcoastlines(ax=mapax)
    plt.show()    

    if pickPoints:
        cid = fig.canvas.mpl_connect('key_press_event', lambda event: onpick3(event, m))

        
def onpick3(event, m):
    print( m(event.xdata, event.ydata, inverse=True))

    
def plot_region_bathy(region=None, wd=7, ht=5, bathy=None, save=False, savename="untitled.png", levs=[]):
    plt.close(1)
    fig = plt.figure(1, figsize=(wd, ht))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 0.02])

    if(region == "CDP"):
        llcrnrlon, llcrnrlat = 60, -68
        urcrnrlon, urcrnrlat = 71, -65.5
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "WPB"):
        llcrnrlon, llcrnrlat = 69, -70
        urcrnrlon, urcrnrlat = 80, -66
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "EPB"):
        llcrnrlon, llcrnrlat = 69, -70
        urcrnrlon, urcrnrlat = 83, -64.5
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "LAC"):
        llcrnrlon, llcrnrlat = 80, -68
        urcrnrlon, urcrnrlat = 89, -64.5
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "KC"):
        llcrnrlon, llcrnrlat = 99, -68
        urcrnrlon, urcrnrlat = 112, -63
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "AC"):
        llcrnrlon, llcrnrlat = 133, -68
        urcrnrlon, urcrnrlat = 147, -64
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "RS"):
        llcrnrlon, llcrnrlat = 160, -79
        urcrnrlon, urcrnrlat = 180, -71
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "AS"):
        llcrnrlon, llcrnrlat = -122, -75.5
        urcrnrlon, urcrnrlat = -98, -70
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "BS"):
        llcrnrlon, llcrnrlat = -102, -71.5
        urcrnrlon, urcrnrlat = -58, -60
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)        
    if(region == "WS"):
        llcrnrlon, llcrnrlat = -50, -82
        urcrnrlon, urcrnrlat = -20, -72
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "PMC"):
        llcrnrlon, llcrnrlat = -19, -74
        urcrnrlon, urcrnrlat = 0, -65
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "PHC"):
        llcrnrlon, llcrnrlat = 28, -71
        urcrnrlon, urcrnrlat = 38, -65
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)                
    else:
        map_proj = ccrs.PlateCarree()
    
    
    mapax = plt.subplot(gs[:,0], projection=map_proj)
    mapax.set_extent([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat], crs=ccrs.PlateCarree())
    colorbar_ax = plt.subplot(gs[1:3,1])

    try:
        if bathy.variables:
            pass
    except:
        bathy = xr.open_dataset(MEOPDIR+'/Datasets/Bathymetry/GEBCO_2014_2D.nc')
    bathyS = bathy.sel(lon=slice(llcrnrlon-0.5, urcrnrlon+0.5), lat=slice(llcrnrlat-0.5, urcrnrlat+0.5))

    cs2 = mapax.contourf(bathyS.lon, bathyS.lat, bathyS.elevation.where(bathyS.elevation <= 0).values,  levels=levs, extend="min", transform=ccrs.PlateCarree()) #, , cmap='rainbow'   , levels=clevs,
    ## plt.figure(2)
    ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
    ## plt.figure(1)
    cbar = Colorbar(ax = colorbar_ax, mappable = cs2, orientation = 'vertical')
    cbar.ax.get_children()[0].set_linewidths(5)
    cbar.set_label('Depth (m)')
    MEOPDIR = "/media/data"
    shpfile = MEOPDIR+"/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6.shp"
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry']) for shp in records]
    mapax.add_geometries(geometries, ccrs.PlateCarree(), edgecolor='gray', facecolor='none')
    
    ISedgefname = MEOPDIR+"/Datasets/Shapefiles/AntIceShelfEdge/ne_10m_antarctic_ice_shelves_lines.shp"
    ISe_feature = ShapelyFeature(Reader(ISedgefname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor="k")
    mapax.add_feature(ISe_feature, zorder=3)

    mapax.set_extent([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat])        
    gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xlabels_top = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': 'k'}
    gl.ylabel_style = {'size': 8, 'color': 'k'}

    if save:
        plt.savefig(savename, dpi=300)

    plt.show()



def plotDataDensity_NIS_DIS(df1, df2, units='Data Density', save=False, savename="savedFig.png", wd=7, ht=7, show=True, mapax = None, subplotlabel=None, levels=[0, 10, 20, 30, 40, 50, 60, 100, 200, 500], region='Whole', plotBathy=True, fontsize=8, DATADIR = "/media/data", dx=0.5, dy=0.5, region_lons=None):

    df = [df1, df2]    
    matplotlib.rcParams.update({'font.size': fontsize})        # setting fontsize for plot elements            
    plt.figure(1, figsize=(wd,ht));
    external_mapaxis = True
    if not mapax:
        print("creating local mapaxis")
        external_mapaxis = False
        gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.03, 0.03])
        mapax = plt.subplot(gs[0, 0], projection = ccrs.Orthographic(central_longitude=0, central_latitude=-90) )
        datacolorbar1 = plt.subplot(gs[0, 1])
        #datacolorbar2 = plt.subplot(gs[0, 2])

    shpfile = DATADIR+"/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6.shp"
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry']) for shp in records]
    ISedgefname = DATADIR+"/Datasets/Shapefiles/AntIceShelf/ne_10m_antarctic_ice_shelves_polys.shp"
    ISe_feature = ShapelyFeature(Reader(ISedgefname).geometries(), 
                                 ccrs.PlateCarree(), linewidth=0.2,
                                 facecolor='none', 
                                 edgecolor="k")

    mapax.add_geometries(geometries, ccrs.PlateCarree(), edgecolor='0.25', facecolor='0.7',alpha=0.25, linewidth=0.2)
    mapax.add_feature(ISe_feature, zorder=3)


    gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, zorder=2,
                  linewidth=0.5, color='gray', alpha=1, linestyle='--')
    
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if region_lons:
        gl_regions = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle=':', zorder=3)
        #gl_regions_lowZ = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle='--', zorder=1)
        gl_regions.xlocator = mticker.FixedLocator(region_lons)
        gl_regions.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5) )
        gl_regions.xformatter = LONGITUDE_FORMATTER
        gl_regions.yformatter = LATITUDE_FORMATTER
        gl_regions.ylines = False
    
    matplotlib.rcParams.update({'font.size': fontsize})    
    
    color_scheme = [['paleturquoise', 'deepskyblue', 'navy'], ['mistyrose', 'salmon', 'darkred']]
    #color_scheme = ['paleturquoise', 'deepskyblue', 'navy']
    CF = []
    for r in range(1):
        surfIndex = df[r].groupby('PROFILE_NUMBER').head(1).index

        xlons, ylats = df[r].loc[surfIndex , 'LONGITUDE'].values, df[r].loc[surfIndex,'LATITUDE'].values

        Xgrid = np.arange(-180, 180.1, dx)
        Ygrid = np.arange(-80, -60.1, dy)
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

        cmap = LinearSegmentedColormap.from_list(name='linearCmap', colors=color_scheme[1], N=len(levels)-1) 

        CF.append(mapax.pcolormesh(XX, YY, ni_masked, vmin=min(levels), vmax=max(levels), norm=bnorm, 
                                   cmap=cmap, transform = ccrs.PlateCarree(), zorder=1))
    
    
    if(plotBathy == True):
        bathyS = xr.open_dataset('/media/hdd2/SOSE_1_12/bathyS.nc')
        cs = mapax.contour(bathyS.lon, bathyS.lat, bathyS.elevation.where(bathyS.elevation <= 0).values,  levels=[-1000], colors="b", linestyle=":", linewidths=0.25, transform = ccrs.PlateCarree())
        
    if not external_mapaxis:
        print("no external mapaxis")
        cbar1 = Colorbar(ax = datacolorbar1, mappable = CF[0], ticks=ticks, orientation='vertical') 
    #cbar2 = Colorbar(ax = datacolorbar2, mappable = CF[1], ticks=ticks, orientation='vertical')
    #cbar1.ax.set_yticklabels("")
    #cbar2.set_label("Data density")

    for l in np.arange(-160, 181, 20):
        if( (l == 80) or (l == 100) ):
            text_lat = -80
        else:
            text_lat  = -62.5
        mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )
    if not subplotlabel:
        pass
    else:
        mapax.text(-134, -49, subplotlabel, transform = ccrs.PlateCarree() )
    if(save==True):
        plt.savefig(savename, dpi=300, bbox_inches='tight')
    if mapax:
        return CF[0]
    else:
        plt.show();

        

def plot_fields_orthographic(field, longitude_coord, latitude_coord, vmin, vmax, save=False, savename="untitled.png",
                             colorbar_label="", region_lons=None, mapax=None, plot_colorbar=True):
    DATADIR = "/media/data"
    passing_mapaxis = True
    if not mapax:
        passing_mapaxis = False
        plt.close(1)
        plt.figure(1, figsize=(7,7) )
        gs = gridspec.GridSpec(5, 2, width_ratios=[1, 0.05], wspace=0.05)
    
        mapax = plt.subplot(gs[:,0], projection = ccrs.Orthographic(central_longitude=0, central_latitude=-90) )
        
    CF = mapax.pcolormesh(longitude_coord, latitude_coord, 
                          field, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin = vmin, vmax = vmax)

    gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, zorder=2,
                  linewidth=0.5, color='gray', alpha=1, linestyle='--')
    
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if region_lons:
        gl_regions = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle=':', zorder=3)
        #gl_regions_lowZ = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle='--', zorder=1)
        gl_regions.xlocator = mticker.FixedLocator(region_lons)
        gl_regions.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5) )
        gl_regions.xformatter = LONGITUDE_FORMATTER
        gl_regions.yformatter = LATITUDE_FORMATTER
        gl_regions.ylines = False

    shpfile = DATADIR+"/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6.shp"
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry']) for shp in records]
    ISedgefname = DATADIR+"/Datasets/Shapefiles/AntIceShelf/ne_10m_antarctic_ice_shelves_polys.shp"
    ISe_feature = ShapelyFeature(Reader(ISedgefname).geometries(), 
                                 ccrs.PlateCarree(), linewidth=0.2,
                                 facecolor='none', 
                                 edgecolor="k")

    mapax.add_geometries(geometries, ccrs.PlateCarree(), edgecolor='0.25', facecolor='0.7',alpha=0.25, linewidth=0.2)
    mapax.add_feature(ISe_feature, zorder=3)

    for l in np.arange(-160, 181, 20):
        if( (l == 80) or (l == 100) ):
            text_lat = -62.5
        else:
            text_lat  = -62.5
        mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )    

    if not passing_mapaxis:
        colorbar_ax = plt.subplot(gs[1:-1, 1])
        Colorbar(mappable = CF, ax = colorbar_ax)
        colorbar_ax.set_ylabel(colorbar_label)

    if save:
        plt.savefig(savefig, dpi=600)

    if passing_mapaxis:
        return CF
    plt.show()
