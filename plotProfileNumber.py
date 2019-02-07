import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy.ma as ma
import mplcursors
import matplotlib.gridspec as gridspec # GRIDSPEC !
from haversine import haversine
import xarray as xr
from matplotlib.colorbar import Colorbar

def plotProfileNumberContours(df,var="PROFILE_NUMBER", plotBathy=True, pickProfs=True, pickPoints=False,
                         save=False, savename="savedFig.png", m=None, pfno_exclude=[], pfno_include=[],
                         wd=7, ht=7, cx=10, cy=10, cmap='viridis', nmin=0, region='Whole', levels=[0], show=True):
    try:
        df.PLATFORM_NUMBER = dfmg["PLATFORM_NUMBER"].apply(lambda x: x.split("'")[1])
    except:
        pass

    plt.close(1)
    fig = plt.figure(1,figsize=(wd,ht));
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    mapax = plt.subplot(gs[0,0])
        
    lat0 = -90
    lon0 = 0

    if not m:
        m  = createMapProjections(lat0, lon0, region=region)
    bathycolorbar = plt.subplot(gs[1, 0])
    m.readshapefile("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6", "GSHHS_f_L6", color='0.5', linewidth=1., ax=mapax)
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
    surfIndex = df.index.isin(df.groupby('PROFILE_NUMBER').tail(1).index)
    excludePfNo = [True] * len(df)
    includePfNo = [True] * len(df)
    if pfno_exclude:
        excludePfNo = ~df.PLATFORM_NUMBER.isin(pfno_exclude)
    if pfno_include:
        includePfNo = df.PLATFORM_NUMBER.isin(pfno_include)
    surfIndex = surfIndex & excludePfNo & includePfNo
    platformNos = df.loc[surfIndex].PLATFORM_NUMBER.unique()
    z = df.loc[surfIndex, var].values
    
    labels = []
    for pfno in platformNos:
        platMask = (df.loc[surfIndex, 'PLATFORM_NUMBER'] == pfno)
        if(len(df.loc[surfIndex].loc[platMask]) > 0):
            xlons, ylats = m(df.loc[surfIndex].loc[platMask , 'LONGITUDE'].values, df.loc[surfIndex].loc[platMask,'LATITUDE'].values)
            profile_time = df.loc[surfIndex].loc[platMask, 'JULD'].values
            profile_no = df.loc[surfIndex].loc[platMask, 'PROFILE_NUMBER'].values
            profile_depth_of_dive = df.loc[surfIndex].loc[platMask, 'DEPTH'].values
            prof_dist_gline = df.loc[surfIndex].loc[platMask, 'DIST_GLINE'].values
            lines =  m.plot(xlons, ylats, 'o', label="Pno= "+str(pfno), ax=mapax)        
            m.plot(xlons, ylats, ax=mapax, color='0.7')

            pfNoLabels = ["Pfno= "+pfno+"\n"] * len(xlons)
            labels = labels+[pno + str(ptime) +"\n"+ str(profno)+"\n"+ str(pdod) +"\n"+str(distGL) for pno, ptime, profno, pdod, distGL in zip(pfNoLabels, profile_time, profile_no, profile_depth_of_dive, prof_dist_gline) ]

    if pickProfs:
        cursor = mplcursors.cursor(lines)
        cursor.connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    if pickPoints:
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onpick3(event, m) )
        print("cid:", cid)
        #fig.canvas.mpl_connect('pick_event', onpick3)        
        #cursor = mplcursors.cursor()
    
    if(plotBathy == True):
        bathy = xr.open_dataset('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-1000, -1500, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = m.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  latlon=True, levels=clevs, linewidths=0.8, extend='min', ax=mapax) #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'horizontal')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')

        
    m.drawcoastlines(ax=mapax)
    if(save==True):
        plt.savefig(savename, dpi=300)
    if(show == True):
        plt.show();
    else:
        plt.close();

    #return lines


def get_bounding_box(m, xllcorner, yllcorner, xurcorner, yurcorner):

    x1, y1 = m(xllcorner, yllcorner, inverse=True)
    y2 = y1
    x2, _ = m(xurcorner, yllcorner, inverse=True)    
    x3 = x2
    _, y3 = m(xurcorner, yurcorner, inverse=True)
    x4 = x1
    y4 = y3

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def get_box_dfselector(df, boundingBox):
    #receive boundingBox as [[llcrnrx, llcrnry], [urcrnrx, urcrnry]]
    try:
        leftlonlim = boundingBox[0][0]
        rightlonlim = boundingBox[1][0]
        lowerlatlim = boundingBox[0][1]
        upperlatlim = boundingBox[1][1]
    except:
        raise ValueError('provide arg boundingBox in fmt [[x1,y1], [x2, y2], [x3, y3], [x4, y4]] \n With x1,y1 being lower left corner of bounding box, and going counter-clockwise from that point.')
    
    selectProfiles = (df.LATITUDE >= lowerlatlim) & (df.LATITUDE <= upperlatlim) & (df.LONGITUDE >= leftlonlim) & (df.LONGITUDE <= rightlonlim) & (~df.CTEMP.isnull())

    return selectProfiles
    

def onpick3(event, m):
    if(event.button == 1):
        print(event.xdata, ",", event.ydata, event.button) #ind, np.take(x, ind), np.take(y, ind))
        print(m(event.xdata, event.ydata, inverse=True), "\n" )
        

