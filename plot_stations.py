import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_station_locations(positions, title=' ', save=False, savename="untitled.png", wd=12, ht=12):
    x = positions[:,1]
    y = positions[:,0]
    
    plt.figure(figsize=(wd,ht));
            
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
    xgrid,ygrid = m(x,y)    
    
    m.drawmapboundary();
    m.drawcoastlines()
    #m.fillcontinents(color='#cc9966');
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='m')
    m.scatter(xgrid,ygrid,marker='+',color='b');
    
    parallels = np.arange(-80, -30+1, 5.)
    
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True]*len(parallels))
    meridians = np.arange(-180, 181, 20.)

    labels = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    import numpy.ma as ma
    ma.masked_array(meridians, mask=labels)
    print(labels)
    m.drawmeridians(meridians,labels=[True]*len(meridians))
    plt.title(title, y=1.07)
    #plt.tight_layout()

    if(save== True):
        plt.savefig(savename)
    plt.show();
    plt.close();
    #for i in range(len(xgrid)):
        #plt.annotate(str(stations[i]), (xgrid[i],ygrid[i]))
