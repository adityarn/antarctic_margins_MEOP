from shapely.geometry import Point
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from haversine import haversine
import geopandas as gpd

dfmg = pd.read_csv("dfmg.csv")
dfmg['DIST_GLINE'] = np.empty(len(dfmg))

grndLine = gpd.read_file("/media/data/Datasets/Shapefiles/AntarcticGroundingLine/GSHHS_f_L6.shp")

def find_distance_to_groundingline(gdf, grndLine):
    indices = gdf.index
    df_lonlat = np.array([[gdf.loc[indices[0], 'LONGITUDE'],  gdf.loc[indices[0], 'LATITUDE'] ]])
    point = Point(df_lonlat[0])
    closest_polygon_ind = np.argmin(grndLine.geometry[:].distance(point))

    poly_points = np.array(list(grndLine.geometry[closest_polygon_ind].exterior.coords))
    closest_gl_point_ind = np.argmin(cdist(poly_points , df_lonlat))
    
    distance = float(haversine(poly_points[closest_gl_point_ind, ::-1], df_lonlat[0, ::-1]))
    print(distance, type(distance))
    gdf.loc[:, 'DIST_GLINE'] = distance
    return gdf



dfmg = dfmg.groupby(['LATITUDE', 'LONGITUDE']).apply(find_distance_to_groundingline, grndLine)

dfmg.to_csv("dfmg_glDist.csv")
