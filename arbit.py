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
