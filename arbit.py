plt.close(1)
plt.figure(1, figsize=(190/25.4, 230/25.4) )

mapax = plt.subplot(projection = ccrs.Orthographic(central_longitude=0, central_latitude=-90) )

DATADIR = "/media/data"

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

sel_inds = dfmg.loc[sel_tags].groupby("PROFILE_NUMBER").head(1).index
lons = dfmg.loc[sel_inds, "LONGITUDE"].unique()
lats = dfmg.loc[sel_inds, "LATITUDE"].unique()


gl_regions = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle=':', zorder=3)
#gl_regions_lowZ = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle='--', zorder=1)
gl_regions.xlocator = mticker.FixedLocator(region_lons)
gl_regions.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5) )
gl_regions.xformatter = LONGITUDE_FORMATTER
gl_regions.yformatter = LATITUDE_FORMATTER
gl_regions.ylines = False


if(plotBathy == True):
    bathyS = xr.open_dataset('/media/hdd2/SOSE_1_12/bathyS.nc')
    cs = mapax.contour(bathyS.lon, bathyS.lat, bathyS.elevation.where(bathyS.elevation <= 0).values,  levels=[-1000], colors="b", linestyle=":", linewidths=0.25, transform = ccrs.PlateCarree())
    
mapax.scatter(lons, lats, marker="x", color="b")
