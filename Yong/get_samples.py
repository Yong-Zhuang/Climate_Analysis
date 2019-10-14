import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


folder = "../data/"
G_CSV_PATH = f"{folder}X_Ganges.csv"
GY_CSV_PATH = f"{folder}Y_Ganges.csv"
B_CSV_PATH = f"{folder}X_Brahmaputra.csv"
M_CSV_PATH = f"{folder}X_Meghna.csv"
RF_CSV_PATH = f"{folder}persiann_1_x_1_look20.npy"
FP_CSV_PATH = f"{folder}total_forecast_precipitation_mean_spread_input.npy"
LAT = np.arange(-19,45,1)
LON = np.arange(60,188,1)

def get_samples(look = 10, lead = 10, normalization = True):
    Qx_Ganges = pd.read_csv(G_CSV_PATH)
    Qx_Ganges.head()

    Qy_Ganges = pd.read_csv(GY_CSV_PATH)
    Qy_Ganges.head()

    Q = pd.concat([Qx_Ganges, Qy_Ganges], axis=1)
    Q.head()
    idx = []
    for i in np.arange(1 - look, lead+1, 1):
        idx.append("Q_" + str(i))

    target_Q = Q.loc[:,idx[1:]]#Q_-8...Q_10

    X_Q = Q.loc[:,idx[:-1]] #Q_-9...Q_9
    if normalization:
        normalizer = StandardScaler()
        norm_num_data = normalizer.fit_transform(X_Q.values)
        X_Q.loc[:,idx[:-1]] = norm_num_data
        
    mb_rf = pd.read_csv(rainfall_save_path)

    rf = np.load(RF_CSV_PATH)#(4896, 20, 64, 128)
    rf = rf.reshape(rf.shape[0], rf.shape[1],-1)
    rf = rf[:,:,mb_rf["idx"].values]
    if normalization:
        for i in range(rf.shape[1]):
            normalizer = StandardScaler()
            norm_num_data = normalizer.fit_transform(rf[:,i,:])
            rf[:,i,:] = norm_num_data    
    
    fp = np.load(FP_CSV_PATH)#(4896, 64, 128, 15, 2)
    fp = np.moveaxis(fp, 3, 1)
    fp = fp[:, :, :, :, 0]#(4896, 15, 64, 128)
    fp = fp.reshape(fp.shape[0], fp.shape[1],-1)
    fp = fp[:,:,mb_rf["idx"].values]
    if normalization:
        for i in range(fp.shape[1]):
            normalizer = StandardScaler()
            norm_num_data = normalizer.fit_transform(fp[:,i,:])
            fp[:,i,:] = norm_num_data     
    X_rf = np.concatenate((rf[:,-look:,:], fp[:,:lead-1,:]), axis=1)#(4896, 19, 689)
    
    input_encoder_streamflow, input_decoder_streamflow = X_Q.iloc[:look].values,X_Q.iloc[look:].values
    input_observed,input_forecasted = X_rf[:,:look,:],X_rf[:,look:,:]
    y_ob_sf,y_fo_sf = target_Q.iloc[:look].values,target_Q.iloc[look:].values
    
    if normalization:
        print("MinMax normalization")
        Pob, Px_mean, Px_spread = Normalization(Pob, Px_mean, Px_spread)
    return input_encoder_streamflow, input_decoder_streamflow, input_observed,input_forecasted, y_ob_sf,y_fo_sf