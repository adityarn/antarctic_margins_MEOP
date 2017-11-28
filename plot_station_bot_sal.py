import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_station_bot_sal(df,var="PSAL_ADJUSTED", elev=[0], lons=[0], lats=[0], title=' ', colorunit='Cond.', cmin=33, cmax=35.5, contour=False,
                         save=False, savename="savedFig.png"):
    #not_null = ~df[var].isnull()
    
    color_var = df.loc[df.groupby('PROFILE_NUMBER').tail(1).index, var].values
    x = df.loc[df.groupby('PROFILE_NUMBER').tail(1).index, 'LONGITUDE'].values
    y = df.loc[df.groupby('PROFILE_NUMBER').tail(1).index, 'LATITUDE'].values
    
    plt.figure(1,figsize=(15,15));
            
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
    lat0 = -89
    lon0 = 0
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='i');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='i', llcrnrx=-width*0.30, llcrnry=-0.30*height,\
                 urcrnrx=0.30*width, urcrnry=0.30*height)
    
    #lat_0 = -60, lon_0 = -20,
    xm,ym = m(x,y)    
    
    m.drawmapboundary();
    m.drawcoastlines()
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='m')
    #m.fillcontinents(color='#ddaa66');
    
    sc = m.scatter(xm,ym, c=color_var, vmin=cmin, vmax=cmax); #, cmap='viridis'
    cbar = m.colorbar(sc, pad="10%", location='bottom') 
    cbar.set_label(colorunit)
    parallels = np.arange(-80, -50+1, 5.)
    
    if(contour==True):
        lon2, lat2 = np.meshgrid(lons,lats) # get lat/lons of ny by nx evenly space grid.
        xx, yy = m(lon2, lat2) # compute map proj coordinates.
    
        levs = np.array([-600, -900, -1500])
        cont = m.contour(xx, yy, elev, levels=levs[::-1],  colors='0.5', linestyles='-', linewidths=(0.5,)) #clevs[::-1], , cmap='rainbow'
        plt.clabel(cont, fmt = '%2.1d', colors = '0.5', fontsize=8)
    
    m.drawparallels(parallels,labels=[True]*len(parallels))
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[True]*len(meridians))
    plt.title(title, y=1.05)
    plt.tight_layout()
    if(save==True):
        plt.savefig(savename, dpi=150)
    plt.show();
    #for i in range(len(xgrid)):
        #plt.annotate(str(stations[i]), (xgrid[i],ygrid[i]))
