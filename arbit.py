def plot_fields_orthographic(field, longitude_coord, latitude_coord, vmin, vmax, save=False, savename="untitled.png",
                             colorbar_label="", region_lons=None):
    DATADIR = "/media/data"
    plt.close(1)
    plt.figure(1, figsize=(7,7) )
    gs = gridspec.GridSpec(5, 2, width_ratios=[1, 0.05], wspace=0.05)

    mapax = plt.subplot(gs[:,0], projection = ccrs.Orthographic(central_longitude=0, central_latitude=-90) )
    CF = mapax.pcolormesh(longitude_coord, latitude_coord, 
                          field, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin = -8, vmax = 8)
    mapax.coastlines()

    gl = mapax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, zorder=2,
                  linewidth=0.5, color='gray', alpha=1, linestyle='--')
    
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-80, -59, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if region_lons:
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

    for l in np.arange(-160, 181, 20):
        if( (l == 80) or (l == 100) ):
            text_lat = -80
        else:
            text_lat  = -62.5
        mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )    
        
    colorbar_ax = plt.subplot(gs[1:-1, 1])
    Colorbar(mappable = CF, ax = colorbar_ax)
    colorbar_ax.set_ylabel(colorbar_label)

    if save:
        plt.savefig(savefig, dpi=600)
    plt.show()
