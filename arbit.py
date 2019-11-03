plt.close(1)
plt.figure(1, figsize=(190/25.4, 230/25.4) )
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.01, 0.1], wspace=0.0, hspace=0)
ax = plt.subplot(gs[0,0], projection = ccrs.Orthographic(central_latitude=-90, central_longitude=0) )
cbar_ax = plt.subplot(gs[0,2])


sel_inds = dfmg.loc[~sel_tags].groupby("PROFILE_NUMBER").head(1).index
lons = dfmg.loc[sel_inds, "LONGITUDE"]
lats = dfmg.loc[sel_inds, "LATITUDE"]
years = dfmg.loc[sel_inds, "JULD"].dt.year
ax.scatter(lons, lats, c="k", marker=".", transform=ccrs.PlateCarree() )

sel_inds = dfmg.loc[sel_tags].groupby("PROFILE_NUMBER").head(1).index
lons = dfmg.loc[sel_inds, "LONGITUDE"]
lats = dfmg.loc[sel_inds, "LATITUDE"]
years = dfmg.loc[sel_inds, "JULD"].dt.year
SC = ax.scatter(lons, lats, c=years, marker=".", transform=ccrs.PlateCarree() )
cbar = plt.colorbar(mappable = SC, cax = cbar_ax)




#ax.set_ylabel("Latitude")
#ax.set_xlabel("Longitude")
#ax.set_xlim(-180, 180)
#ax.set_xticks(np.arange(-180, 181, 60))

gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, zorder=2,
              linewidth=0.5, color='gray', alpha=1, linestyle='--')

gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
gl.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# if region_lons:
#     gl_regions = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle=':', zorder=3)
#     #gl_regions_lowZ = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=1, linestyle='--', zorder=1)
#     gl_regions.xlocator = mticker.FixedLocator(region_lons)
#     gl_regions.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5) )
#     gl_regions.xformatter = LONGITUDE_FORMATTER
#     gl_regions.yformatter = LATITUDE_FORMATTER
#     gl_regions.ylines = False

matplotlib.rcParams.update({'font.size': 8})    

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

for l in np.arange(-160, 181, 20):
    if( (l == 80) or (l == 100) ):
        text_lat = -80
    else:
        text_lat  = -62.5
    mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )

cbar.set_label("Years")

plt.savefig("./Images/dataDensity/anomalousSalinityTagsLocation.jpg", dpi=600, bbox_inches="tight")
