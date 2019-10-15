import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


folder = "../Data/"
G_CSV_PATH = str(folder)+"X_Ganges.csv"
GY_CSV_PATH = str(folder)+"Y_Ganges.csv"
B_CSV_PATH = str(folder)+"X_Brahmaputra.csv"
M_CSV_PATH = str(folder)+"X_Meghna.csv"
RF_CSV_PATH = str(folder)+"persiann_1_x_1_look20.npy"
FP_CSV_PATH = str(folder)+"total_forecast_precipitation_mean_spread_input.npy"
RF_MB_CSV_PATH = "./markov_blanket_for_rainfall.csv.gz"
FP_MB_CSV_PATH = "./markov_blanket_for_rainfall_g_1.csv.gz"
LAT = np.arange(-19,45,1)
LON = np.arange(60,188,1)

def get_samples(look = 10, lead = 10, sdim = 5, normalization = True):
    Qx_Ganges = pd.read_csv(G_CSV_PATH)

    Qy_Ganges = pd.read_csv(GY_CSV_PATH)

    Q = pd.concat([Qx_Ganges, Qy_Ganges], axis=1)
    idx = []
    for i in np.arange(1 - look, lead+1, 1):
        idx.append("Q_" + str(i))
   
    cols = [col for col in Q.columns if 'Q_' in col]
    target_Q = Q.loc[:,idx[1:]].copy()#Q_-8...Q_10

    if normalization:
        normalizer = StandardScaler()

        norm_num_data = normalizer.fit_transform(Q[cols].values)
        Q.loc[:,cols] = norm_num_data
        
    mb_rf = pd.read_csv(RF_MB_CSV_PATH)
    mb_fp = pd.read_csv(FP_MB_CSV_PATH)

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
    X = np.zeros((Q.shape[0],len(idx[:-1]),sdim))
    
    
    #X_Q = Q.loc[:,idx[:-1]] 
    for i,v in enumerate(idx[:-1]):#Q_-9...Q_9
        vs = v.split("_")
        today = int(vs[1])
        for j in range(sdim): 
            day = today - j
            col = "Q_"+str(day)
            X[:,i,j] = Q.loc[:,col]
    #X_Q = X_Q.values.reshape(X_Q.shape[0],X_Q.shape[1],1)
    target_Q = target_Q.values.reshape(target_Q.shape[0],target_Q.shape[1],1)
    print(X.shape)
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    input_encoder_streamflow, input_decoder_streamflow = X[:,:look,:],X[:,look:,:]
    input_observed,input_forecasted = X_rf[:,:look,:],X_rf[:,look:,:]
    y_ob_sf,y_fo_sf = target_Q[:,:look],target_Q[:,look:]
    
    return input_encoder_streamflow, input_decoder_streamflow, input_observed,input_forecasted, y_ob_sf,y_fo_sf
