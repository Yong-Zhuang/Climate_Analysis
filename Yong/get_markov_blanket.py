import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import analytics.utils.feature_selection as fs


if __name__== "__main__": 
    parser = argparse.ArgumentParser(description='castle')    
    parser.add_argument('-diff', metavar='int', required=True, help='0: use diff; 1: no') 
    parser.add_argument('-river', metavar='str', required=True, help='g: Ganges; b: Brahmaputra; m: Meghna') 
    parser.add_argument('-norm', metavar='int', required=True, help='0: normalization diff; 1: no') 
    args = parser.parse_args() 
    diff = int(args.diff)
    norm = int(args.norm)
    folder = "./Data/"
    G_CSV_PATH = f"{folder}X_Ganges.csv"
    B_CSV_PATH = f"{folder}X_Brahmaputra.csv"
    M_CSV_PATH = f"{folder}X_Meghna.csv"
    if args.river == "g":
        River_CSV_PATH = f"{folder}X_Ganges.csv"
    elif args.river == "b":
        River_CSV_PATH = f"{folder}X_Brahmaputra.csv"
    else
        River_CSV_PATH = f"{folder}X_Meghna.csv"
        
    RF_CSV_PATH = f"{folder}persiann_1_x_1_look20.npy"
    FP_CSV_PATH = f"{folder}total_forecast_precipitation_mean_spread_input.npy"
    LAT = np.arange(-19,45,1)
    LON = np.arange(60,188,1)

    Qx_Ganges = pd.read_csv(River_CSV_PATH)
    Qx_Ganges.head()
    Qx = Qx_Ganges.loc[:,["Q_-1","Q_0"]]
    Qx["diff"] = Qx["Q_0"] - Qx["Q_-1"]
    if diff == 0:
        target = Qx["diff"]
    else:
        target = Qx["Q_0"]

    rf = np.load(RF_CSV_PATH)#(4896, 20, 64, 128)
    rf = rf[:, -1, :, :] 

    coor = []
    for i in LAT:
        for j in LON:
            coor.append(f"{i},{j}")

    rf= rf.reshape(rf.shape[0],-1)
    if norm==0:
        for i in range(rf.shape[1]):
            normalizer = StandardScaler()
            norm_num_data = normalizer.fit_transform(rf[:,i])
            rf[:,i] = norm_num_data 
    df = pd.DataFrame(rf, columns = coor)


    rainfall_save_path = "markov_blanket_for_rainfall_Ganges1.csv.gz"
    if os.path.exists(rainfall_save_path):
        mb_rf = pd.read_csv(rainfall_save_path)
        display(mb_rf)
    else:
        model = fs.MRMRMB()
        mb_rf = model.get_mb(df, target, prob=0.95)
        mb_rf.to_csv(rainfall_save_path, index=None, header=True, compression="gzip")