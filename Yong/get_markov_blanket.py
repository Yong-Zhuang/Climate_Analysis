import os
import pandas as pd
import numpy as np
import analytics.utils.feature_selection as fs

folder = "./data/"
G_CSV_PATH = f"{folder}X_Ganges.csv"
B_CSV_PATH = f"{folder}X_Brahmaputra.csv"
M_CSV_PATH = f"{folder}X_Meghna.csv"
RF_CSV_PATH = f"{folder}persiann_1_x_1_look20.npy"
FP_CSV_PATH = f"{folder}total_forecast_precipitation_mean_spread_input.npy"
LAT = np.arange(-19,45,1)
LON = np.arange(60,188,1)


Qx_Ganges = pd.read_csv(G_CSV_PATH)
Qx_Ganges.head()
Qx = Qx_Ganges.loc[:,["Q_-1","Q_0"]]
Qx["diff"] = Qx["Q_0"] - Qx["Q_-1"]
target = Qx["diff"]

rf = np.load(RF_CSV_PATH)#(4896, 20, 64, 128)
rf = rf[:, -1, :, :] 

coor = []
for i in LAT:
    for j in LON:
        coor.append(f"{i},{j}")
        
rf= rf.reshape(rf.shape[0],-1)
df = pd.DataFrame(rf, columns = coor)
df.head()

rainfall_save_path = "markov_blanket_for_rainfall_Ganges.csv.gz"
if os.path.exists(rainfall_save_path):
    mb_rf = pd.read_csv(rainfall_save_path)
    display(mb_rf)
else:
    model = fs.MRMRMB()
    mb_rf = model.get_mb(df, target, prob=0.95)
    mb_rf.to_csv(rainfall_save_path, index=None, header=True, compression="gzip")