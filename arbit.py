for l in np.arange(0, 360, 20):
    if( (l == 80) or (l == 100) ):
        text_lat = -80
    else:
        text_lat  = -62.5
    mapax.text(l, text_lat, str(l)+"$^{\circ}$", transform=ccrs.PlateCarree() )
