def compute_stress_curl(winds):
    stress_curl = np.zeros_like(winds.iews)
    wek = np.zeros_like(winds.iews)
    rho0 = 1025.0
    tauy_x = np.zeros_like(winds.iews)
    taux_y = np.zeros_like(winds.iews)
    
    delta_lat = abs(float(winds.latitude[0] - winds.latitude[1]))
    delta_lon = abs(float(winds.longitude[0] - winds.longitude[1]))
    r = 6371e3 # radius of earth in metres
    dy = float(r * np.deg2rad(delta_lat))
    print(dy)
    omega = 2*np.pi/(24.*3600)
    f = 2. * omega * np.sin(np.deg2rad(winds.latitude.values))
    time_counter = 0
    for t in winds.time:
    
        lat_counter = 0
        for lat in winds.latitude:
            
            dx = float(r * np.cos(np.deg2rad(lat) ) * np.deg2rad(delta_lon))
            
            tauy_x[time_counter, lat_counter] = np.gradient(winds.sel(time = t, latitude=lat).inss , dx)
            lat_counter+=1
        taux_y[time_counter], _ = np.gradient(winds.sel(time = t).iews , -dy, dx)
        stress_curl[time_counter] = tauy_x[time_counter] - taux_y[time_counter]
        
        for l in range(len(winds.latitude)):
            wek[time_counter, l] = stress_curl[time_counter, l] / (f[l] * rho0)
        time_counter += 1
    
    print(dx)
    windEk = xr.Dataset({'stressCurl':(['time', 'latitude', 'longitude'], stress_curl), 
                         'wek':(['time', 'latitude', 'longitude'], wek) }, 
                        coords={'time': winds.time, 'latitude': winds.latitude, 
                               'longitude': winds.longitude})
    return windEk



def compute_clim_stress_curl_the_WRONG_way(winds_monthlyMean):
    stress_curl = np.zeros_like(winds_monthlyMean.iews)
    wek = np.zeros_like(winds_monthlyMean.iews)
    rho0 = 1025.0
    tauy_x = np.zeros_like(winds_monthlyMean.iews)
    taux_y = np.zeros_like(winds_monthlyMean.iews)
    
    delta_lat = abs(float(winds_monthlyMean.latitude[0] - winds_monthlyMean.latitude[1]))
    delta_lon = abs(float(winds_monthlyMean.longitude[0] - winds_monthlyMean.longitude[1]))
    r = 6371e3 # radius of earth in metres
    dy = float(r * np.deg2rad(delta_lat))
    print(dy)
    omega = 2*np.pi/(24.*3600)
    f = 2. * omega * np.sin(np.deg2rad(winds_monthlyMean.latitude.values))
    time_counter = 0
    for m in winds_monthlyMean.month:
    
        lat_counter = 0
        for lat in winds_monthlyMean.latitude:
            
            dx = float(r * np.cos(np.deg2rad(lat) ) * np.deg2rad(delta_lon))
            
            tauy_x[time_counter, lat_counter] = np.gradient(winds_monthlyMean.sel(month = m, latitude=lat).inss , dx)
            lat_counter+=1
        taux_y[time_counter], _ = np.gradient(winds_monthlyMean.sel(time = t).iews , -dy, dx)
        stress_curl[time_counter] = tauy_x[time_counter] - taux_y[time_counter]
        
        for l in range(len(winds_monthlyMean.latitude)):
            wek[time_counter, l] = stress_curl[time_counter, l] / (f[l] * rho0)
        time_counter += 1
    
    print(dx)
    windEk = xr.Dataset({'stressCurl':(['time', 'latitude', 'longitude'], stress_curl), 
                         'wek':(['time', 'latitude', 'longitude'], wek) }, 
                        coords={'time': winds_monthlyMean.time, 'latitude': winds_monthlyMean.latitude, 
                               'longitude': winds_monthlyMean.longitude})
    return windEk
