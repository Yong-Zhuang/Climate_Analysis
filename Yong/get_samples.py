import os
import pandas as pd
import numpy as np


folder = "./data/"
G_CSV_PATH = f"{folder}X_Ganges.csv"
GY_CSV_PATH = f"{folder}Y_Ganges.csv"
B_CSV_PATH = f"{folder}X_Brahmaputra.csv"
M_CSV_PATH = f"{folder}X_Meghna.csv"
RF_CSV_PATH = f"{folder}persiann_1_x_1_look20.npy"
FP_CSV_PATH = f"{folder}total_forecast_precipitation_mean_spread_input.npy"
LAT = np.arange(-19,45,1)
LON = np.arange(60,188,1)


Qx_Ganges = pd.read_csv(G_CSV_PATH)
Qx_Ganges.head()

Qy_Ganges = pd.read_csv(GY_CSV_PATH)
Qy_Ganges.head()

Q = pd.concat([Qx_Ganges, Qy_Ganges], axis=1)
Q.head()

look = 10
lead = 10
idx = []
for i in np.arange(1 - look, lead+1, 1):
    idx.append("Q_" + str(i))
    
target_Q = Q.loc[:,idx[1:]]
target_Q.head()

X_Q = Q.loc[:,idx[:-1]]
X_Q.shape, X_Q.head()

mb_rf = pd.read_csv(rainfall_save_path)

rf = np.load(RF_CSV_PATH)#(4896, 20, 64, 128)
rf = rf.reshape(rf.shape[0], rf.shape[1],-1)
rf = rf[:,:,mb_rf["idx"].values]
fp = np.load(FP_CSV_PATH)#(4896, 64, 128, 15, 2)
fp = np.moveaxis(fp, 3, 1)
fp = fp[:, :, :, :, 0]#(4896, 15, 64, 128)
fp = fp.reshape(fp.shape[0], fp.shape[1],-1)
fp = fp[:,:,mb_rf["idx"].values]

X_rf = np.concatenate((rf[:,-look:,:], fp[:,:lead-1,:]), axis=1)
X_rf.shape