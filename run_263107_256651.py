import numpy as np
import pandas as pd
import TBS_fullyCoupled_branch1 as TBS
import importlib
import linecache
importlib.reload(TBS)
import xarray as xr
import os

case_str = "case_263107_256651"
U10 = xr.open_dataset("/media/data/Datasets/ESRL_U10/uwnd.10m.gauss.2011.nc")
V10 = xr.open_dataset("/media/data/Datasets/ESRL_U10/vwnd.10m.gauss.2011.nc")
casefile = "case_263107_256651.txt" #"case_256581_256582.txt" #

outdir = "./Results/arbit/"+str(case_str)
os.system("mkdir "+str(outdir))

def get_U10_value(U10, lat, lon, time):
    latind = np.asscalar(np.argmin(np.abs(U10['lat'] - lat)))
    lonind = np.asscalar(np.argmin(np.abs(U10['lon'] - (lon+180))))
    timeind = np.where(U10['time'].values.astype('<M8[D]') == time)[0][0]

    return np.asscalar(U10['uwnd'][timeind][latind][lonind])


lat = float(linecache.getline(casefile, 1).split()[2])
lon = float(linecache.getline(casefile, 2).split()[2])
time = np.datetime64(linecache.getline(casefile, 3).split()[2])
initial_data = pd.read_csv(casefile, "\t", skiprows=3)
initial_data = initial_data.rename(columns=lambda x: x.strip())
depth = initial_data["depth"].values
dp_x = initial_data["dp_x"].values
dp_y = initial_data["dp_y"].values
rho = initial_data["rho"].values

time = np.datetime64('2011-03-21')
latind = np.asscalar(np.argmin(np.abs(U10['lat'].values - lat)))
lonind = np.asscalar(np.argmin(np.abs(U10['lon'] - (lon+180))))
timeind = np.where(U10['time'].values.astype('<M8[D]') == time)[0][0]
u10 = np.asscalar(U10['uwnd'][timeind][latind][lonind])

time = np.datetime64('2011-03-21')
latind = np.asscalar(np.argmin(np.abs(V10['lat'] - lat)))
lonind = np.asscalar(np.argmin(np.abs(V10['lon'] - (lon+180))))
timeind = np.where(V10['time'].values.astype('<M8[D]') == time)[0][0]
v10 = np.asscalar(V10['vwnd'][timeind][latind][lonind])


#dp_y[0] *= 1e3
#dp_y[-1] *= 1e3
Model = TBS.Keps_Solver(lat, depth, dp_x, dp_y, rho, u10,v10, outdir, "Vars", init_u=[0.1,0.1], init_v=[0.1,0.1], mode='verbose', dy1_bot=0.01, dy1_surf=0.5, z0_bot=0.001, case=case_str, dy=2.5)
Model.solver(36000, 1.0, 300)
