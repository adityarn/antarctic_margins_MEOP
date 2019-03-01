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
from mpldatacursor import datacursor
import cartopy.crs as ccrs
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import re

def plotProfileNumberContours(df,var="PROFILE_NUMBER", plotBathy=True, pickProfs=True, pickPoints=False,
                              save=False, savename="savedFig.png", bathy=None, MEOPDIR="/media/data", include_profnos=[], dod_maxlim=-350, exclude_profnos=[],
                         wd=7, ht=7, cx=10, cy=10, cmap='viridis', nmin=0, region='Whole', levels=[0], show=True):
    try:
        df.PLATFORM_NUMBER = dfmg["PLATFORM_NUMBER"].apply(lambda x: x.split("'")[1])
    except:
        pass

    plt.close(1)
    fig = plt.figure(1,figsize=(wd,ht));
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    mapax = plt.subplot(gs[0,0], projection= ccrs.PlateCarree() )
        
    lat0 = -90
    lon0 = 0
    

    if(region == "CDP"):
        llcrnrlon, llcrnrlat = 65, -68
        urcrnrlon, urcrnrlat = 70, -66
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "WPB"):
        llcrnrlon, llcrnrlat = 69, -70
        urcrnrlon, urcrnrlat = 80, -66
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    if(region == "AS"):
        llcrnrlon, llcrnrlat = -120, -75.5
        urcrnrlon, urcrnrlat = -101, -70
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
    if(region == "WS"):
        llcrnrlon, llcrnrlat = -50, -82
        urcrnrlon, urcrnrlat = -20, -72
        map_proj = ccrs.PlateCarree() #ccrs.Orthographic(65, -90)
    else:
        map_proj = ccrs.PlateCarree()


    if include_profnos:
        surfIndex = df.index.isin(df.groupby('PROFILE_NUMBER').tail(1).index) & df.PROFILE_NUMBER.isin(include_profnos) & (df.DEPTH < dod_maxlim) & ~df.CTEMP.isnull()
        z = df.loc[surfIndex, var].values
    elif exclude_profnos:
        surfIndex = df.index.isin(df.groupby('PROFILE_NUMBER').tail(1).index) & ~df.PROFILE_NUMBER.isin(exclude_profnos) & (df.DEPTH < dod_maxlim) & ~df.CTEMP.isnull()
        z = df.loc[surfIndex, var].values
    else:
        surfIndex = df.index.isin(df.groupby('PROFILE_NUMBER').tail(1).index) & (df.DEPTH < dod_maxlim) & ~df.CTEMP.isnull()
        z = df.loc[surfIndex, var].values
        
    
    scat_collections = []
    if(len(df.loc[surfIndex]) > 0):
        xlons, ylats = df.loc[surfIndex, 'LONGITUDE'].values, df.loc[surfIndex,'LATITUDE'].values
        profile_time = df.loc[surfIndex, 'JULD'].values
        profile_no = df.loc[surfIndex, 'PROFILE_NUMBER'].values
        profile_depth_of_dive = df.loc[surfIndex, 'DEPTH'].values
        prof_nans = np.isnan(profile_depth_of_dive)
        prof_dist_gline = df.loc[surfIndex, 'DIST_GLINE'].values
        ctemp = df.loc[surfIndex, 'CTEMP'].values
        for i in range(len(df[surfIndex])):
            #scat_collections.append(plt.scatter(xlons[i], ylats[i], c="gray", label='profno={}'.format(profile_no[i])+'\n dod={}'.format(profile_depth_of_dive[i])+'\n dist={}'.format(prof_dist_gline[i])+'\n CT={}'.format(df.loc[surfIndex].CTEMP.values[i])+'$,\circ$C') )
            scat_collections.append(plt.scatter(xlons[i], ylats[i], c="gray", label='profno={}'.format(profile_no[i])) )
            datacursor(scat_collections[i], draggable=True)

        mapax.scatter(xlons[~prof_nans], ylats[~prof_nans], c=profile_depth_of_dive[~prof_nans], vmin=-1000, vmax=dod_maxlim)
        
    if pickPoints:
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onpick3(event, m) )
        print("cid:", cid)
        #fig.canvas.mpl_connect('pick_event', onpick3)        
        #cursor = mplcursors.cursor()
        
    bathycolorbar = plt.subplot(gs[1, 0])
    GLfname = MEOPDIR+"/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6.shp"
    GL_feature = ShapelyFeature(Reader(GLfname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor="0.5")
    mapax.add_feature(GL_feature, zorder=3)

    ISedgefname = MEOPDIR+"/Datasets/Shapefiles/AntIceShelfEdge/ne_10m_antarctic_ice_shelves_lines.shp"
    ISe_feature = ShapelyFeature(Reader(ISedgefname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor="k")
    mapax.add_feature(ISe_feature, zorder=3)
    
    if(plotBathy == True):
        try:
            if bathy.variables:
                pass
        except:
            bathy = xr.open_dataset(MEOPDIR+'/Datasets/Bathymetry/GEBCO_2014_2D.nc')
        lonlen = len(bathy.lon)
        lonindices = np.arange(0, lonlen+1, 30)
        lonindices[-1] = lonindices[-1] - 1
        bathyS = bathy.isel(lon=lonindices, lat=np.arange(0, 3600, 5))
        clevs = np.array([-1000, -1500, -2000, -3000])[::-1]
        
        longrid, latgrid = np.meshgrid(bathyS.lon.values, bathyS.lat.values)
        cs = mapax.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  levels=clevs, linewidths=0.8, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        cbar1 = Colorbar(ax = bathycolorbar, mappable = cs, orientation = 'horizontal')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')
    print(llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)
    mapax.set_extent([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat] )
    #mapax.coastlines()
    gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
    fig.canvas.mpl_connect('button_press_event', lambda event: onpick3(event, scat_collections) )

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
    

def onpick3(event, scat_collections):
    for sc in scat_collections:
        if sc.contains(event)[0]:
            print(re.findall('(profno=)(\d+)', sc.get_label())[0][1], end=", " )        

