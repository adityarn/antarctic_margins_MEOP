
# # coding: utf-8

# # # MEOP dataset analysis
# # MEOP data can be found at: http://www.meop.net/database/format.html
# # MEOP stands for: "Marine Mammals Exploring the Oceans from Pole to Pole".
# # MEOP consists of ARGO like profiles, however, the instruments are fitted onto marine mammals.
# # The UK dataset of MEOP contains many profiles in the Weddel Sea region, the following analysis is of this dataset.

# # In[29]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw
from netCDF4 import Dataset
import xarray as xr
from scipy.interpolate import griddata


# Make sure that all the paths to the netcdf files are given in a column in the file named "filenames.txt"
# 
# This can be done by 
# 1. navigating to the folder in the terminal and typing in: "ls -1 | grep .nc > filenames.txt"
# 2. copy and paste "filenames.txt" into the working folder of the Jupyter Notebook
# 3. append the full path name to the file list by:  cat filenames.txt | xargs -i echo "/path/to/ncARGO/{}" > filenames.txt
# 

# In[5]:

filenames = pd.read_csv("filenames.txt", header=None) 


# In[6]:

#func to read in all the files in the filenames
def read_all_nc(filenames):
    data_array = [i for i in range(len(filenames))]
    i=0
    for i in range(len(filenames)):
        data_array[i] = xr.open_dataset(filenames[0][i])
    return data_array      


# In[7]:

# func to create a merged pandas dataframe of all the netcdf files

def create_merged_df(filenames):
    for i in range((len(filenames))):
        arr_data = xr.open_dataset(filenames[i])
        nlev = len(arr_data['PRES_ADJUSTED'][0])
        nprof = len(arr_data['PRES_ADJUSTED'])
        if(i == 0):
            NPROF = np.arange(nprof)
            NPROF = np.array([[NPROF[j]] * nlev for j in range(len(NPROF)) ])
        else:
            last_end = NPROF.flatten()[-1] + 1
            NPROF = np.arange(last_end, last_end+nprof )
            NPROF = np.array([[NPROF[j]] * nlev for j in range(len(NPROF)) ])
        
        ind = np.arange(nprof*nlev)
        lat = np.array([[arr_data["LATITUDE"][j].values]* nlev for j in range(nprof) ])
        lon = np.array([[arr_data["LONGITUDE"][j].values]* nlev for j in range(nprof) ])
        posqc = np.array([[arr_data["POSITION_QC"][j].values]* nlev for j in range(nprof) ])
        juld = np.array([[arr_data["JULD"][j].values]* nlev for j in range(nprof) ])
        
        if(i == 0):    
            df = {'PLATFORM_NUMBER': pd.Series([str(arr_data["PLATFORM_NUMBER"][0].values)]*len(ind), index=ind), 
                  'PROFILE_NUMBER' : pd.Series(NPROF.flatten(), index=ind ),
                  'TEMP_ADJUSTED': pd.Series(arr_data["TEMP_ADJUSTED"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED': pd.Series(arr_data["PSAL_ADJUSTED"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED': pd.Series(arr_data["PRES_ADJUSTED"].values.flatten(), index=ind),  
                  'PRES_ADJUSTED_QC': pd.Series(arr_data["PRES_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED_ERROR':pd.Series(arr_data["PRES_ADJUSTED_ERROR"].values.flatten(), index=ind), 
                  'TEMP_ADJUSTED_QC': pd.Series(arr_data["TEMP_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'TEMP_ADJUSTED_ERROR': pd.Series(arr_data["TEMP_ADJUSTED_ERROR"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED_QC': pd.Series(arr_data["PSAL_ADJUSTED_QC"].values.flatten(), index=ind),  
                  'PSAL_ADJUSTED_ERROR':pd.Series(arr_data["PSAL_ADJUSTED_ERROR"].values.flatten(), index=ind),  
                  'JULD': pd.Series(juld.flatten(), index=ind),  
                  'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                  'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                  'POSITION_QC': pd.Series(posqc.flatten(), index=ind) }
            df = pd.DataFrame(df)
        else:
            df_i = {'PLATFORM_NUMBER': pd.Series([str(arr_data["PLATFORM_NUMBER"][0].values)]*len(ind), index=ind), 
                  'PROFILE_NUMBER' : pd.Series(NPROF.flatten(), index=ind ),
                  'TEMP_ADJUSTED': pd.Series(arr_data["TEMP_ADJUSTED"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED': pd.Series(arr_data["PSAL_ADJUSTED"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED': pd.Series(arr_data["PRES_ADJUSTED"].values.flatten(), index=ind),  
                  'PRES_ADJUSTED_QC': pd.Series(arr_data["PRES_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED_ERROR':pd.Series(arr_data["PRES_ADJUSTED_ERROR"].values.flatten(), index=ind), 
                  'TEMP_ADJUSTED_QC': pd.Series(arr_data["TEMP_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'TEMP_ADJUSTED_ERROR': pd.Series(arr_data["TEMP_ADJUSTED_ERROR"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED_QC': pd.Series(arr_data["PSAL_ADJUSTED_QC"].values.flatten(), index=ind),  
                  'PSAL_ADJUSTED_ERROR':pd.Series(arr_data["PSAL_ADJUSTED_ERROR"].values.flatten(), index=ind),  
                  'JULD': pd.Series(juld.flatten(), index=ind),  
                  'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                  'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                  'POSITION_QC': pd.Series(posqc.flatten(), index=ind)  }
            df_i = pd.DataFrame(df_i)
            
            #pdb.set_trace()
            df = df.append(df_i, ignore_index=True)
            del(arr_data)
    return df

# In[8]:
print("BEginning reading in nc files")
dataAUS = read_all_nc(filenamesAUS)
dfmAUS = create_merged_df(dataAUS)
del(dataAUS)


# In[10]:

dataBRA = read_all_nc(filenamesBRA)
dfmBRA = create_merged_df(dataBRA)
del(dataBRA)


# In[32]:

dataCHN = read_all_nc(filenamesCHN)
dfmCHN = create_merged_df(dataCHN)
del(dataCHN)


# In[12]:

dataFRA = read_all_nc(filenamesFRA)
dfmFRA = create_merged_df(dataFRA)
del(dataFRA)


# In[13]:

dataGER = read_all_nc(filenamesGER)
dfmGER = create_merged_df(dataGER)
del(dataGER)


# In[14]:

dataNOR = read_all_nc(filenamesNOR)
dfmNOR = create_merged_df(dataNOR)
del(dataNOR)


# In[15]:

dataSA = read_all_nc(filenamesSA)
dfmSA = create_merged_df(dataSA)
del(dataSA)


# In[16]:

dataUK = read_all_nc(filenamesUK)
dfmUK = create_merged_df(dataUK)
del(dataUK)


# In[17]:

dataUSA = read_all_nc(filenamesUSA)
dfmUSA = create_merged_df(dataUSA)
del(dataUSA)


# In[26]:
print("BEginning concatenating all nc files")
alldata = [dfmAUS, dfmBRA, dfmCHN, dfmFRA, dfmGER, dfmNOR, dfmSA, dfmUK, dfmUSA]
dfm = pd.concat(alldata, ignore_index=True)


# In[31]:

del(dfmAUS, dfmBRA, dfmCHN, dfmFRA, dfmGER, dfmNOR, dfmSA, dfmUK, dfmUSA)


# Uncomment the cell below if you want to read in the entire dataframe that you may have already written out into a csv file. Reading in the csv is **faster** than converting from netcdf to pandas dataframe

# In[2]:

# QC flags are stored as bytes, converting them to string here
# 1. Flag: 1 = Good data
# 2. Flag: 4 = Bad data
# 3. Flag: 9 = Null value

# In[33]:

dfm['TEMP_ADJUSTED_QC'] = dfm['TEMP_ADJUSTED_QC'].str.decode("utf-8")
dfm['PRES_ADJUSTED_QC'] = dfm['PRES_ADJUSTED_QC'].str.decode("utf-8")
dfm['PSAL_ADJUSTED_QC'] = dfm['PSAL_ADJUSTED_QC'].str.decode("utf-8")


# In[3]:

dfm["JULD"] = pd.to_datetime(dfm['JULD'],infer_datetime_format=True)


# In[4]:

dfm["DEPTH"] = gsw.z_from_p(dfm["PRES_ADJUSTED"], dfm["LATITUDE"])


# uncomment the cell below if you want the entire "dfm" dataframe to be written out into a csv file

# #### Using Gibbs Sea Water Equation of State (TEOS-10) functions, we calculate the density and the conserved density/temperature

SA = gsw.SA_from_SP(dfm['PSAL_ADJUSTED'],dfm['PRES_ADJUSTED'],dfm['LONGITUDE'],dfm['LATITUDE'])
CT = gsw.CT_from_t(SA, dfm['TEMP_ADJUSTED'], dfm['PRES_ADJUSTED'])
dfm['DENSITY_INSITU'] = gsw.density.rho(SA, CT, dfm['PRES_ADJUSTED'])

dfm['POT_DENSITY'] = gsw.density.sigma0(SA, CT)
dfm['CTEMP'] = CT
dfm['SA'] = SA

mask_notbad_temp = ~(dfm['TEMP_ADJUSTED_QC'] == 4)
mask_notbad_sal = ~(dfm['PSAL_ADJUSTED_QC'] == 4)
mask_notbad_pres = ~(dfm['PRES_ADJUSTED_QC'] == 4)

dfmg = dfm[mask_notbad_pres & mask_notbad_sal & mask_notbad_temp] # data with only good QC + null value flags

from scipy.io import netcdf
bathy = netcdf.netcdf_file('/media/data/Datasets/Bathymetry/GEBCO_2014_2D.nc', 'r')

lons = bathy.variables['lon'][:]
lats = bathy.variables['lat'][:]
lonbounds = [-180, 180] # creating lon,lat bounds for region of interest, ie. Weddel Sea
latbounds = [-80, -60]

latli = np.argmin( np.abs(lats - latbounds[0]) ) # extracting the indices of start and end of the region of interest
latui = np.argmin( np.abs(lats - latbounds[1]) )
lonli = np.argmin( np.abs(lons - lonbounds[0]) )
lonui = np.argmin( np.abs(lons - lonbounds[1]) )
elev_below60S = bathy.variables['elevation'][latli:latui, lonli:lonui] # elevation of SO below 60S
mask_below0 = elev_below60S[:,:] > 0
elev_below60S = np.ma.masked_array(elev_below60S[:,:] , mask_below0)


below60S = dfm["LATITUDE"] < -60
unique_positions = dfm.loc[below60S,"LATITUDE": "LONGITUDE"].drop_duplicates(subset=['LONGITUDE', 
                                                                               'LATITUDE']).values
lons = lons[lonli:lonui]
lats = lats[latli:latui]


# In[34]:

longrid, latgrid = np.meshgrid(lons[lonli:lonui], lats[latli:latui])


# In[35]:

xycoords = np.column_stack((longrid.ravel(), latgrid.ravel()))
xycoords.shape

del(longrid, latgrid)


# Below operation is computationally intesive, **be careful with size of input data. Ensure sufficient RAM**
# 
# This operation is carried out once, echodepth is written into dataframe, and then written out into csv file
# 
# 2nd time onwards, read in the csv file

from scipy import spatial

print("BEginning to build tree")
tree = spatial.KDTree(xycoords)

print("BEginnning query")
rowindices = np.array(tree.query(unique_positions[:], k=1)[1] / len(lons), dtype=int)
colindices = np.array(tree.query(unique_positions[:], k=1)[1] % len(lons))
elev_unique_positions = np.copy(elev_below60S[rowindices, colindices])

del(tree, rowindices, colindices, xycoords)

grouped = dfm[below60S].groupby(['LATITUDE', 'LONGITUDE'])

dfm['ECHODEPTH'] = np.zeros(len(dfm))

print("Beginning to assign echodepth")
def assign_echodepth(up):
    for i in range(len(up)):
        dfm.loc[grouped.get_group(tuple(up[i])).index, 'ECHODEPTH'] = elev_unique_positions[i]
        if(i%1000 == 0):
            print(i)
        
assign_echodepth(unique_positions)

del(grouped, elev_unique_positions)

dfm.to_csv("ncARGOmerged_withechodepth.csv")
