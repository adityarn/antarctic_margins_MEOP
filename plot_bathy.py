import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib

def createMapProjections(lat0, lon0, region='Whole', fontsize=14):
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
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='h', llcrnrx=-width*0.15, llcrnry=0.05*height, urcrnrx=0.*width, urcrnry=0.18*height)
    elif(region == 'Ross'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=180, resolution='h', llcrnrx=-width*0.075, llcrnry=height*0.08, urcrnrx=width*0.03, urcrnry=height*0.18)
    elif(region == 'Prydz'):
        m = Basemap(projection='ortho', lat_0=lat0, lon_0=75, resolution='h', llcrnrx=-width*0.1, llcrnry=height*0.12, urcrnrx=width*0.07, urcrnry=height*0.225)
    elif(region == 'Global'):
        m = m1

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.2)
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=0.2)    

    parallels = np.arange(-80, -50+1, 5.)    
    m.drawparallels(parallels,labels=[1,0,0,0], linewidth=0.2) # labels: left,right,top,bottom
    meridians = np.arange(-180, 180, 10.)
    m.drawmeridians(meridians,labels=[1,1,1,1], linewidth=0.2)

    xy = [[-140, -55], [-140, -60] , [-140, -65], [-140, -70], [-140, -75] , [-140, -80]]
    xytext = np.arange(55, 81, 5)
    for i in range(len(xytext)):
        mxy = m(xy[i][0], xy[i][1])
        plt.annotate(str(xytext[i])+"$^o$S", xy=(mxy[0], mxy[1]), rotation=-45, fontsize=12)

    return m


def plot_bathy(bathy, title=' ', wd=12, ht=12, save=False, savename="untitled.png", region="Whole", show=True, fontsize=14):
    matplotlib.rcParams.update({'font.size': fontsize})        
    plt.figure(1,figsize=(wd,ht));
    width=1e7
    height=7e6
    
    lat0 = -90
    lon0 = 0
    
    m  = createMapProjections(lat0, lon0, region=region)    
    
    longrid, latgrid = np.meshgrid(bathy.lon.values, bathy.lat.values)

    clevs = np.array([0,-100,-200,-300,-400,-800, -1000, -1500, -2000, -3000,-6000, -9000])[::-1]
    
    cs = m.contourf(longrid, latgrid, bathy.elevation.where(bathy.elevation <= 0).values,  latlon=True, levels=clevs, vmin=clevs[0], vmax=clevs[-1]) #, , cmap='rainbow'   , levels=clevs,
    cbar = m.colorbar(cs, pad="3%", location='bottom') 
    cbar.set_label('Depth (m)')
    
    plt.title(title, y=1.05)
    plt.tight_layout()

    if(save == True):
        plt.savefig(savename, dpi=150)
    if(show == True):
        plt.show()
