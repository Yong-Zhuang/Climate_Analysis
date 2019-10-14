import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from utils.feature_selection import MRMRMB as mrmrmb


if __name__== "__main__": 
    parser = argparse.ArgumentParser(description='castle')    
    parser.add_argument('-diff', metavar='int', required=True, help='0: use diff; 1: no') 
    parser.add_argument('-river', metavar='str', required=True, help='g: Ganges; b: Brahmaputra; m: Meghna') 
    parser.add_argument('-norm', metavar='int', required=True, help='0: normalization; 1: no')
    parser.add_argument('-rf',metavar='int',required=True,help='0:observed rainfall; 1: rainfall forecasts') 
    args = parser.parse_args() 
    diff = int(args.diff)
    norm = int(args.norm)
    isrf = int(args.rf)
    folder = "../Data/"
    if river == "g":
        FLOW_X_CSV_PATH = = str(folder)+"X_Ganges.csv"
        FLOW_Y_CSV_PATH = = str(folder)+"Y_Ganges.csv"
    elif args.river == "b":
        FLOW_X_CSV_PATH = str(folder)+"X_Brahmaputra.csv"
        FLOW_Y_CSV_PATH = str(folder)+"Y_Brahmaputra.csv"
    else:
        FLOW_X_CSV_PATH = str(folder)+"X_Meghna.csv"
        FLOW_Y_CSV_PATH = str(folder)+"Y_Meghna.csv"
        
    RF_OBSERVED_CSV_PATH = str(folder)+"persiann_1_x_1_look20.npy"
    RF_PREDICTED_CSV_PATH = str(folder)+"total_forecast_precipitation_mean_spread_input.npy"
    LAT = np.arange(-19,45,1)
    LON = np.arange(60,188,1)
    Qx = pd.read_csv(FLOW_X_CSV_PATH)
    Qy = pd.read_csv(FLOW_Y_CSV_PATH)
    Q = pd.concat([Qx,Qy],axis = 1)
    Q = Q.loc[:,["Q_-1","Q_0","Q_1","Q_2"]]
    Q["diff_0"] = Q["Q_1"] - Q["Q_0"]
    Q["diff_1"] = Q["Q_2"] - Q["Q_1"]
    if isrf==0:
        rf = np.load(RF_OBSERVED_CSV_PATH)#(4896, 20, 64, 128)
        rf = rf[:, -1, :, :]   
        if diff == 0:
            target = Qx["diff_0"]
        else:
            target = Qx["Q_1"]
    else:    
        rf = np.load(RF_PREDICTED_CSV_PATH)
        rf = np.moveaxis(rf, 3, 1)
        rf = rf[:, :, :, :, 0]#(4896, 15, 64, 128)
        rf = rf[:,0,:,:]  
        if diff == 0:
            target = Qx["diff_1"]
        else:
            target = Qx["Q_2"]  


    coor = []
    for i in LAT:
        for j in LON:
            coor.append(str(i)+","+str(j))

    rf= rf.reshape(rf.shape[0],-1)
    if norm==0:
        for i in range(rf.shape[1]):
            normalizer = StandardScaler()
            norm_num_data = normalizer.fit_transform(rf[:,i:i+1])
            rf[:,i] = norm_num_data[:,0]
    df = pd.DataFrame(rf, columns = coor)


    rainfall_save_path = "markov_blanket_for_rainfall_"+args.river+"_"+args.diff+"_"+args.rf+".csv.gz"
    if os.path.exists(rainfall_save_path):
        mb_rf = pd.read_csv(rainfall_save_path)
        display(mb_rf)
    else:
        model = mrmrmb()
        mb_rf = model.get_mb(df, target, prob=0.95)
        mb_rf.to_csv(rainfall_save_path, index=None, header=True, compression="gzip")
