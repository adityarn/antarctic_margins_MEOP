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
import cartopy
import fiona
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.ticker as ticker

def plot_AcrossASF(acASF, dfmg, windsDat, windEk, wd=7, ht=4, savefig=False, savename="untitled.png", levels=[], xlat=False, show=True, MEOPDIR = "/media/data", p2pdist=True, fontsize=8,
                   region='Whole', bathy=None, plotBathy=True, llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=+180, urcrnrlat=+90):
    matplotlib.rcParams.update({'font.size': fontsize})    
    plt.close(1)
    fig = plt.figure(1, figsize=(wd, ht))
    gs = gridspec.GridSpec(8, 2, height_ratios=[0.01, 1,0.01, 1.5, 0.01, 0.26, 0.26, 0.26], width_ratios=[1, 0.02], hspace=0.6, wspace=0.05)
    contour_ax = plt.subplot(gs[3,0])

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
        
    mapax = plt.subplot(gs[5:8,0], projection=map_proj)
    mapax.set_extent([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat], crs=ccrs.PlateCarree())
    colorbar_ax1 = plt.subplot(gs[3,1])
    colorbar_ax2 = plt.subplot(gs[6:8,1])
    
    profs = acASF.PROFILE_NUMBERS
    profs = [int(x) for x in profs.split(',')]
    year = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[0]]), "JULD"].dt.year.values[0]
    month = dfmg.loc[dfmg.PROFILE_NUMBER.isin([profs[0]]), "JULD"].dt.month.values[0]
    region = acASF.REGION

    ctemps = []
    latlons = []
    depth = []
    gamman = []
    echodepth = []
    dist_gline = []
    zonal = []
    merid = []
    stress_curl = []
    wek = []
    
    no_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(len(profs)):
        dfSelect = dfmg.PROFILE_NUMBER.isin([profs[i]])
        ctemps = np.concatenate((ctemps, dfmg.loc[dfSelect, "CTEMP"].values))
        latlons.append([dfmg.loc[dfSelect, 'LATITUDE'].values[0], dfmg.loc[dfSelect, 'LONGITUDE'].values[0] ])
        wind_lons = dfmg.loc[dfSelect, 'LONGITUDE'].values[0]
        if wind_lons < 0:
            wind_lons = wind_lons + 360
        stress_curl.append(windEk.stressCurl.sel(time=str(year)+"-{0:-02d}".format(month)+"-01", latitude=dfmg.loc[dfSelect, 'LATITUDE'].values[0], longitude=wind_lons, method='nearest') )
        wek.append(windEk.sel(time=slice(str(year)+"-{0:-02d}".format(month)+"-01", str(year)+"-{0:-02d}".format(month)+"-"+str(no_of_days[month-1]))).mean(dim=["time"]).wek.sel(latitude=dfmg.loc[dfSelect, 'LATITUDE'].values[0], longitude= wind_lons, method='nearest') )
        zonal.append(windsDat.sel(time=slice(str(year)+"-{0:-02d}".format(month)+"-01", str(year)+"-{0:-02d}".format(month)+"-"+str(no_of_days[month-1]))).mean(dim=["time"]).u10.sel(latitude=dfmg.loc[dfSelect, 'LATITUDE'].values[0], longitude= wind_lons, method='nearest') )
        merid.append(windsDat.sel(time=slice(str(year)+"-{0:-02d}".format(month)+"-01", str(year)+"-{0:-02d}".format(month)+"-"+str(no_of_days[month-1]))).mean(dim=["time"]).v10.sel(latitude=dfmg.loc[dfSelect, 'LATITUDE'].values[0], longitude= wind_lons, method='nearest') )
        depth = np.concatenate((depth,dfmg.loc[dfSelect, "DEPTH"].values))
        gamman = np.concatenate((gamman, dfmg.loc[dfSelect, "gamman"].values))
        echodepth = np.concatenate((echodepth, [dfmg.loc[dfSelect].ECHODEPTH.values[0]]))

        
        if xlat:
            dist_gline = np.concatenate((dist_gline, dfmg.loc[dfSelect, "LATITUDE"].values))
        elif p2pdist:
            if(i == 0):
                dist_gline = np.concatenate((dist_gline, np.zeros(len(dfmg[dfSelect])) ))
            else:
                dist = dist_gline[-1] + haversine(latlons[i-1], latlons[i])
                dist_gline = np.concatenate( (dist_gline, np.zeros(len(dfmg[dfSelect]), dtype=float) + dist))
        else:
            dist_gline = np.concatenate((dist_gline, dfmg.loc[dfSelect, "DIST_GLINE"].values))
    latlons = np.array(latlons)

    dist_echo = np.unique(dist_gline)
    print(gamman.shape)
    if xlat:
        ndist = int(np.max(dist_gline) / 0.1)
    else:
        ndist = int(np.max(dist_gline) / 2.)
    
    dist_grid = np.linspace(np.min(dist_gline), np.max(dist_gline), ndist)
    ndepth = int(-np.min(depth) / 10.)
    depth_grid = np.linspace(np.min(depth), 0, ndepth)
    
    dist_grid, depth_grid = np.meshgrid(dist_grid, depth_grid)
    gamman_interpolated = griddata(np.array([dist_gline, depth]).T, gamman, (dist_grid, depth_grid), method='cubic' )
    
    cs = contour_ax.scatter(dist_gline, depth, c=ctemps, vmin=-2.0, vmax=0.5, s=10)
    
    slope_labels = np.zeros(len(echodepth), dtype=int)
    slope_labels[echodepth > -1000] = 0
    slope_labels[(echodepth < -1000) & (echodepth > -1500) ] = 1
    slope_labels[(echodepth < -1500) & (echodepth > -2000) ] = 2
    slope_labels[(echodepth < -2000) & (echodepth > -3000) ] = 3
    slope_labels[(echodepth < -3000) ] = 4

    contour_ax_twinx = contour_ax.twiny()
    contour_ax_twinx.set_xticks(dist_echo)
    contour_ax_twinx.set_xticklabels(slope_labels)
        
            
    

    contour_ax.set_ylabel("Depth (m)")
    if xlat:
        contour_ax.set_xlabel("Latitude")
    elif p2pdist:
        contour_ax.set_xlabel("Chainage (km)")
    else:
        contour_ax.set_xlabel("Distance from GL (km)")
    xaxislength = np.max(dist_gline) - np.min(dist_gline)
    contour_ax.set_xlim(np.min(dist_gline) - xaxislength*0.02, np.max(dist_gline) + xaxislength* 0.02)
    contour_ax_twinx.set_xlim(np.min(dist_gline) - xaxislength*0.02, np.max(dist_gline) + xaxislength* 0.02)
        
    if levels:
        cs_gamman = contour_ax.contour(dist_grid, depth_grid, gamman_interpolated, levels=levels, colors='0.5')
    else:
        try:
            levels = np.array(str.split(acASF.LEVELS, ","), dtype=float)
            cs_gamman = contour_ax.contour(dist_grid, depth_grid, gamman_interpolated, levels=levels, colors='0.5')
        except:
            levels = None
            cs_gamman = contour_ax.contour(dist_grid, depth_grid, gamman_interpolated, colors='0.5')

        
    contour_ax.clabel(cs_gamman, colors='k', fontsize=8, fmt='%3.2f')

    #mapax.plot(latlons.LONGITUDE.values, latlons.LATITUDE.values, marker="x", markersize=20, transform=ccrs.PlateCarree() )
    mapax.scatter(latlons[:, 1], latlons[:, 0], marker="x", s=20, transform=ccrs.PlateCarree() )    
    mapax.coastlines()
    parallels = np.arange(-80, -50+1, 5.)    
    #m.drawparallels(parallels,labels=[1,0,0,0], linewidth=0.2, ax=mapax) # labels: left,right,top,bottom
    meridians = np.arange(-180, 180, 20.)
    #m.drawmeridians(meridians,labels=[0,1,1,1], linewidth=0.2, ax=mapax)
    
    cbar1 = Colorbar(ax = colorbar_ax1, mappable = cs, orientation = 'vertical', extend='both', label="CT$^\circ$C")
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
        cs2 = mapax.contour(longrid, latgrid, bathyS.elevation.where(bathyS.elevation <= 0).values,  levels=clevs, linewidths=0.8, transform=ccrs.PlateCarree()) #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(2)
        ## cf = plt.contourf(longrid, latgrid,bathyS.elevation.where(bathyS.elevation <= 0).values, levels=clevs, extend='min') #, , cmap='rainbow'   , levels=clevs,
        ## plt.figure(1)
        cbar1 = Colorbar(ax = colorbar_ax2, mappable = cs2, orientation = 'vertical')
        cbar1.ax.get_children()[0].set_linewidths(5)
        cbar1.set_label('Depth (m)')
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

    gl = contour_ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
    #fig.subplots_adjust(wspace=0.5)
    #fig.subplots_adjust(hspace=1)
    winds_ax = plt.subplot(gs[1, 0])
    lns1 = winds_ax.plot(dist_echo, zonal, ".", label="u", color='b')
    lns2 = winds_ax.plot(dist_echo, merid, "x", label="v", color='b')
    winds_twinax = winds_ax.twinx()
    lns3 = winds_twinax.plot(dist_echo, wek, "+", label="w$_{\mathrm{Ek}}$", color='tab:orange')

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]


    winds_ax.set_xlim(np.min(dist_gline) - xaxislength*0.02, np.max(dist_gline) + xaxislength* 0.02)
    winds_ax.set_xticklabels([])
    winds_ax.set_xticks([])
    winds_twinax.set_xlim(np.min(dist_gline) - xaxislength*0.02, np.max(dist_gline) + xaxislength* 0.02)
    winds_twinax.set_xticklabels([])
    winds_twinax.set_xticks([])
    #winds_ax.set_ylim(-10, 10)
    winds_twinax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2e'))
    winds_ax.set_ylabel("u, v (m/s)")
    winds_twinax.set_ylabel("w$_{\mathrm{Ek}}$ (m/s)")
    #winds_twinax.set_ylim(-2e-6, 2e-6)
    #winds_ax.set_title("o : u,  x : v,  + : w$_{\mathrm{Ek}}$")
    
    space_ax = plt.subplot(gs[4, :], frameon=False)
    space_ax.set_xticks([])
    space_ax.set_xticklabels([])
    space_ax.set_yticks([])
    space_ax.set_yticklabels([])    

    space_ax2 = plt.subplot(gs[2, :], frameon=False)
    space_ax2.set_xticks([])
    space_ax2.set_xticklabels([])
    space_ax2.set_yticks([])
    space_ax2.set_yticklabels([])    
    
    
    title_ax = plt.subplot(gs[0, :], frameon=False)
    title_ax.set_xticks([])
    title_ax.set_xticklabels([])
    title_ax.set_yticks([])
    title_ax.set_yticklabels([])    
    title_ax.set_title(region+", "+str(year)+"/ "+str(month))
    title_ax.legend(lns, labs, loc=4, ncol=3, bbox_to_anchor=(0.95, -1.5))    
    #plt.tight_layout()
    if savefig:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()








def display_patches_ticklabels():
    ytickspace = 0
    S0_ticks = np.array([dist_echo[echodepth>-1000], [ytickspace]*len(dist_echo[echodepth>-1000])] ).T
    S1_ticks = np.array([dist_echo[(echodepth < -1000) & (echodepth > -1500)], [ytickspace]*len(dist_echo[(echodepth < -1000) & (echodepth > -1500)])] ).T
    S2_ticks = np.array([dist_echo[(echodepth < -1500) & (echodepth > -2000)], [ytickspace]*len(dist_echo[(echodepth < -1500) & (echodepth > -2000)])] ).T
    S3_ticks = np.array([dist_echo[(echodepth < -2000) & (echodepth > -3000)], [ytickspace]*len(dist_echo[(echodepth < -2000) & (echodepth > -3000)])] ).T
    S4_ticks = np.array([dist_echo[(echodepth < -3000)], [ytickspace]*len(dist_echo[(echodepth < -3000)])] ).T

    xaxislen =  np.max(dist_gline)+20 - (np.min(dist_gline)-20)
    ytickspace = 50
    for i in range(len(S0_ticks)):
        lowerCorner = contour_ax.transData.transform((S0_ticks[i][0] - 0.05*xaxislen, S0_ticks[i][1] - ytickspace))
        upperCorner = contour_ax.transData.transform((S0_ticks[i][0] + 0.05*xaxislen, S0_ticks[i][1] + ytickspace))
        print([lowerCorner[0], lowerCorner[1], upperCorner[0], upperCorner[1] ])
        bbox_image = BboxImage(Bbox([ [lowerCorner[0], lowerCorner[1]], [upperCorner[0], upperCorner[1]] ]), norm = None, origin=None, clip_on=False)
        bbox_image.set_data(plt.imread('./Images/acrossASF/markers/cross.png'))
        contour_ax.add_artist(bbox_image)        

    for i in range(len(S1_ticks)):
        lowerCorner = contour_ax.transData.transform((S1_ticks[i][0] - 0.05*xaxislen, S0_ticks[i][1] - ytickspace))
        upperCorner = contour_ax.transData.transform((S1_ticks[i][0] + 0.05*xaxislen, S0_ticks[i][1] + ytickspace) )       
        bbox_image = BboxImage(Bbox([ [lowerCorner[0], lowerCorner[1]], [upperCorner[0], upperCorner[1]] ]), norm = None, origin=None, clip_on=False)
        bbox_image.set_data(plt.imread('./Images/acrossASF/markers/circle.png'))
        contour_ax.add_artist(bbox_image)

    for i in range(len(S2_ticks)):
        lowerCorner = contour_ax.transData.transform((S2_ticks[i][0] - 0.05*xaxislen, S0_ticks[i][1] - ytickspace))
        upperCorner = contour_ax.transData.transform((S2_ticks[i][0] + 0.05*xaxislen, S0_ticks[i][1] + ytickspace) )       
        bbox_image = BboxImage(Bbox([ [lowerCorner[0], lowerCorner[1]], [upperCorner[0], upperCorner[1]] ]), norm = None, origin=None, clip_on=False)
        bbox_image.set_data(plt.imread('./Images/acrossASF/markers/square.png'))
        contour_ax.add_artist(bbox_image)

    for i in range(len(S3_ticks)):
        lowerCorner = contour_ax.transData.transform((S3_ticks[i][0] - 0.05*xaxislen, S0_ticks[i][1] - ytickspace))
        upperCorner = contour_ax.transData.transform((S3_ticks[i][0] + 0.05*xaxislen, S0_ticks[i][1] + ytickspace) )       
        bbox_image = BboxImage(Bbox([ [lowerCorner[0], lowerCorner[1]], [upperCorner[0], upperCorner[1]] ]), norm = None, origin=None, clip_on=False)
        bbox_image.set_data(plt.imread('./Images/acrossASF/markers/triangle.png'))
        contour_ax.add_artist(bbox_image)        
                
    for i in range(len(S4_ticks)):
        lowerCorner = contour_ax.transData.transform((S4_ticks[i][0] - 0.05*xaxislen, S0_ticks[i][1] - ytickspace))
        upperCorner = contour_ax.transData.transform((S4_ticks[i][0] + 0.05*xaxislen, S0_ticks[i][1] + ytickspace))        
        bbox_image = BboxImage(Bbox([ [lowerCorner[0], lowerCorner[1]], [upperCorner[0], upperCorner[1]] ]), norm = None, origin=None, clip_on=False)
        bbox_image.set_data(plt.imread('./Images/acrossASF/markers/plus.png'))
        contour_ax.add_artist(bbox_image)
