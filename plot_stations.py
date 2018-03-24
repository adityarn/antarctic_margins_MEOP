import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import mpld3
from mpld3 import plugins

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
    
    m1 = Basemap(projection='ortho', lat_0 =lat0, lon_0 =lon0, resolution='l');
    width = m1.urcrnrx - m1.llcrnrx
    height = m1.urcrnry - m1.llcrnry
    dist_x = 8199.491128244028
    w_factor = 0.3
    h_factor = 0.3
    if(region =='Whole'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='l', llcrnrx=-width*w_factor, llcrnry=-height*h_factor, urcrnrx=w_factor*width, urcrnry=h_factor*height)
    elif(region == 'Weddell'):
        m  = Basemap(projection='ortho', lat_0=lat0, lon_0=lon0, resolution='l', llcrnrx=-width*0.225, llcrnry=-0.0*0.1*height, urcrnrx=0.*width, urcrnry=0.20*height)
    else:
        raise Exception('region should be either \'Whole\' or \'Weddell\' ')

    m.drawmapboundary();
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.75', linewidth=0.1)
    #m.fillcontinents(color='#ddaa66');
    m.drawcoastlines(linewidth=0.2)    

    parallels = np.arange(-80, -50+1, 5.)    
    m.drawparallels(parallels,labels=[1,1,1,0], linewidth=0.2)
    meridians = np.arange(-180, 180, 20.)
    m.drawmeridians(meridians,labels=[1,1,0,1], linewidth=0.2)

    return m

def plot_station_locations_with_mouseover(positions, title=' ', save=False, savename="untitled.png", wd=12, ht=12, region='Whole', markers=None):
    css = """
    table
    {
      border-collapse: collapse;
    }
    th
    {
      color: #ffffff;
      background-color: #000000;
    }
    td
    {
      background-color: #cccccc;
    }
    table, th, td
    {
      font-family:Arial, Helvetica, sans-serif;
      border: 1px solid black;
      text-align: right;
    }
    """    
    x = positions[:,1]
    y = positions[:,0]
    
    fig = plt.figure(figsize=(wd,ht));
    ax = plt.axes()
    lat0 = -90
    lon0 = 0
    m = createMapProjections(lat0, lon0, region=region)    

    xgrid,ygrid = m(x,y)    
    
    m.drawmapboundary();
    m.drawcoastlines()
    #m.fillcontinents(color='#cc9966');
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='m')

    ## if(markers != None):
    ##     for i in range(len(markers)):
    ##         plt.text(xgrid[i],ygrid[i], str(markers[i]), color='b');
    
    plt.title(title, y=1.07)
    parallels = np.arange(-80, -30+1, 5.)
    labels = [1,1,1,0]
    m.drawparallels(parallels,labels=labels)
    meridians = np.arange(-180, 180, 20.)
    labels = [1,1,0,1]
    m.drawmeridians(meridians, labels=labels)

    points_with_annotation = []
    for i in range(len(markers)):
        scat, = m.plot(xgrid[i],ygrid[i],'o',color='b');
        annotation = ax.annotate("%s" % markers[i],
            xy=(xgrid[i], ygrid[i]), xycoords='data',
            xytext=(xgrid[i], ygrid[i]), textcoords='data',
            horizontalalignment="left",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"),
            bbox=dict(boxstyle="round", facecolor="w", 
                      edgecolor="0.5", alpha=0.9)
            )
        # by default, disable the annotation visibility
        annotation.set_visible(False)

        points_with_annotation.append([scat, annotation])


    def on_move(event):
        visibility_changed = False
        for scat, annotation in points_with_annotation:
            should_be_visible = (scat.contains(event)[0] == True)

            if should_be_visible != annotation.get_visible():
                visibility_changed = True
                annotation.set_visible(should_be_visible)

        if visibility_changed:        
            plt.draw()

    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)


    
    if(save== True):
        plt.savefig(savename)

    plt.show();


def plot_station_locations(positions, title=' ', save=False, savename="untitled.png", wd=12, ht=12, region='Whole'):
    x = positions[:,1]
    y = positions[:,0]
    
    fig = plt.figure(figsize=(wd,ht));
    ax = plt.axes()
    lat0 = -90
    lon0 = 0
    m = createMapProjections(lat0, lon0, region=region)    

    xgrid,ygrid = m(x,y)    
    
    m.drawmapboundary();
    m.drawcoastlines()
    #m.fillcontinents(color='#cc9966');
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='m')

    ## if(markers != None):
    ##     for i in range(len(markers)):
    ##         plt.text(xgrid[i],ygrid[i], str(markers[i]), color='b');
    
    parallels = np.arange(-80, -30+1, 5.)
    labels = [1,1,1,0]
    m.drawparallels(parallels,labels=labels)
    meridians = np.arange(-180, 180, 20.)
    labels = [1,1,0,1]
    m.drawmeridians(meridians, labels=labels)

    m.scatter(x, y, latlon=True)

    if(save == True):
        plt.savefig(savename)
    plt.show()
