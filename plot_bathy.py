import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_bathy(bathy, lons, lats, title=' ', wd=12, ht=12, save=False, savename="untitled.png"):
    plt.figure(1,figsize=(wd,ht));
    width=1e7
    height=7e6
    #m = Basemap(projection='hammer',lon_0=270)
    # plot just upper right quadrant (corners determined from global map).
    # keywords llcrnrx,llcrnry,urcrnrx,urcrnry used to define the lower
    # left and upper right corners in map projection coordinates.
    # llcrnrlat,llcrnrlon,urcrnrlon,urcrnrlat could be used to define
    # lat/lon values of corners - but this won't work in cases such as this
    # where one of the corners does not lie on the earth.
    #m1 = Basemap(projection='ortho',lon_0=-9,lat_0=60,resolution=None)
    
    lat0 = -89
    lon0 = 0
    
    #m = Basemap(projection='cyl', llcrnrlon=lons[0], llcrnrlat=lats[0], 
    #            urcrnrlon=lons[-1], urcrnrlat=lats[-1], lat_0 =lat0, lon_0 =lon0, resolution='h');
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='i');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='i', llcrnrx=-width*0.30, llcrnry=-0.30*height,\
                 urcrnrx=0.30*width, urcrnry=0.30*height)
    #m = Basemap(projection='nsper', satellite_height=3000000., llcrnrx=0, llcrnry=0, 
    #            urcrnrx=2.7e6, urcrnry=5e5, lat_0 =lat0, lon_0 =lon0, resolution='l');
    #lat_0 = -60, lon_0 = -20,
    #lons,lats = np.meshgrid(lons,lats)
    #x,y = m(lons, lats)
    
    lon2, lat2 = np.meshgrid(lons,lats) # get lat/lons of ny by nx evenly space grid.
    xx, yy = m(lon2, lat2) # compute map proj coordinates.
    
    
    #x = np.linspace(0, m.urcrnrx, bathy.shape[1])
    #y = np.linspace(0, m.urcrnry, bathy.shape[0])

    #xx, yy = np.meshgrid(x, y)
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((1, 0, 0), (0, 0, 1)), N=10, gamma=10.0)
    clevs = np.array([0,-100,-200,-300,-400,-800, -1500, -3000,-6000, -9000])
    cs = m.contourf(xx, yy, bathy, clevs[::-1]) #clevs[::-1], , cmap='rainbow'
    
    cbar = m.colorbar(cs, pad="3%", location='bottom') 
    cbar.set_label('Depth (m)')
    
      
    #m.drawmapboundary(fill_color='#99ffff');
    m.drawcoastlines();
    m.fillcontinents(color='1.',lake_color='gray');
    #m.scatter(xgrid,ygrid,marker='D',color='m');
    
    parallels = np.arange(-80, -29, 5.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[0, 1, 0, 1])
    meridians = np.arange(int(lons[0]), lons[-1] + 1, 15.)
    m.drawmeridians(meridians,labels=[1, 0, 1, 0])
    plt.title(title, y=1.05)
    plt.tight_layout()

    if(save == True):
        plt.savefig(savename, dpi=150)
    plt.show()
    
    plt.close()
