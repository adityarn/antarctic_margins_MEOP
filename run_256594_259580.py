import numpy as np
import pandas as pd
import TBS_fullyCoupled as TBS
import importlib
import linecache
importlib.reload(TBS)
import xarray as xr
U10 = xr.open_dataset("/media/data/Datasets/ESRL_U10/uwnd.10m.gauss.2011.nc")
V10 = xr.open_dataset("/media/data/Datasets/ESRL_U10/vwnd.10m.gauss.2011.nc")
case_str = "case_256594_259580"
casefile = "./casefiles/"+case_str+".txt"
outdir = "./Results/"+case_str+"_500lvls/"

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

latind = np.asscalar(np.argmin(np.abs(U10['lat'].values - lat)))
lonind = np.asscalar(np.argmin(np.abs(U10['lon'] - (lon+180))))
timeind = np.where(U10['time'].values.astype('<M8[D]') == time)[0][0]
u10 = np.asscalar(U10['uwnd'][timeind][latind][lonind])

latind = np.asscalar(np.argmin(np.abs(V10['lat'] - lat)))
lonind = np.asscalar(np.argmin(np.abs(V10['lon'] - (lon+180))))
timeind = np.where(V10['time'].values.astype('<M8[D]') == time)[0][0]
v10 = np.asscalar(V10['vwnd'][timeind][latind][lonind])


#dp_y[0] *= 1e3
#dp_y[-1] *= 1e3
Model = TBS.Keps_Solver(300, lat, depth, dp_x, dp_y, rho, u10, v10, outdir, "Vars", init_u=[0.01,0.1], init_v=[0.01,0.1], mode='verbose', dy1_bot=0.05, dy1_surf=0.2, z0_bot=0.005, case=case_str)
Model.solver(36000, 1.0, 300)
