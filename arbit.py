DATADIR = "/media/data"
#region1 : all COLD regime regions, color them BLUE
region1LonLims = [ [-40.,-20.] ,  [60,70], [70,75], 
                  [75,82], [135,145], [170,180]]
region1LatLims = [ [-77., -72], [-68., -66.47], [-69.75, -66.35],
                   [-69.75, -65.215], [-67.25, -65.4], [-78., -70]]
titles1 = ["WS", "CD", "WPB",        
           "EPB","AC", "RS"] 


region2LonLims = [ [-20., 0.] ,   [82,87], [101,112], ]
region2LatLims = [ [-72.5, -69],  [-67., -65.56], [-67., -64.5], ]
titles2 = ["PMC",  "LAC", "KC"]

region3LonLims = [[29,37], [-120,-100], [-100,-62.5]]
region3LatLims = [[-70.5, -65.38], [-75.25, -70.7], [-74., -61.4] ]
titles3 = ["PHC", "AS", "BS"]

regionsLonlim = [region1LonLims, region2LonLims, region3LonLims]
regionsLatlim = [region1LatLims, region2LatLims, region3LatLims]

plt.figure(1, figsize=(10, 10) )
mapax = plt.subplot(projection = ccrs.Orthographic(central_latitude=-90, central_longitude=0) )

gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, zorder=2,
              linewidth=0.5, color='gray', alpha=1, linestyle='--')

gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
gl.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

region_lons = [-120, -100, -60, -20, 0, 29, 37, 60, 70, 75, 82, 87, 101, 112, 135, 145, 160, 180]
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


shelf_sel = ~np.isnan(bathyS.where((bathyS.elevation < 0) & (bathyS.elevation > -1000) ).elevation.values)
#cmap = ['Blues', 'cool', 'Reds']
vmin = [-1000, -500, -2000]
vmax = [0, 1000, -500]
from matplotlib.colors import BoundaryNorm
#bnorm = BoundaryNorm(np.arange(), ncolors=5, clip=False)

cmap1 = LinearSegmentedColormap.from_list(name='linearCmap', colors=['blue', 'mediumblue'], N=5)
cmap2 = LinearSegmentedColormap.from_list(name='linearCmap', colors=['aqua', 'cyan'], N=5)
cmap3 = LinearSegmentedColormap.from_list(name='linearCmap', colors=['orangered', 'red'], N=5)
cmap = [cmap1, cmap2, cmap3]

for i in range(len(regionsLonlim) ):
    for j in range(len(regionsLonlim[i])):
        selected_bathyS = bathyS.where(shelf_sel).sel(lon = slice(regionsLonlim[i][j][0], regionsLonlim[i][j][1]), lat = slice(regionsLatlim[i][j][0], regionsLatlim[i][j][1]) )
        mapax.pcolormesh(selected_bathyS.lon, selected_bathyS.lat, selected_bathyS.elevation, vmin=vmin[i], vmax=vmax[i], cmap=cmap[i], transform = ccrs.PlateCarree())
        
for l in np.arange(-160, 181, 20):
    if( (l == 80) or (l == 100) ):
        text_lat = -62.5
    else:
        text_lat  = -62.5
    mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )    

        
plt.savefig("./Images/dataDensity/shelfRegimes.jpg", dpi=600)
