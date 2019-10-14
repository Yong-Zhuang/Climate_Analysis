import numpy as np
import scipy
import scipy.stats
import sys
import argparse
import operator
import pandas as pd
#from __future__ import print_function
import os
import math
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from datetime import timedelta
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from keras.models import model_from_json
from keras import backend as K
from sklearn import metrics
import get_samples as gs
import castle
#from keras.utils import multi_gpu_model


np.random.seed(1337)  # for reproducibility
nb_epoch = 1000  # number of epoch at training stage
batch_size = 100  # batch size
train_test = 3825

def get_metrics(results, lead,y, pred):
    out = pd.DataFrame(data={"y": y, "pred": pred})
    print(out.T)
    m_mae = metrics.mean_absolute_error(y, pred)
    m_rmse = metrics.mean_squared_error(y, pred) ** 0.5
    m_r2 = metrics.r2_score(y, pred)
    results[lead]["MAE"] += m_mae
    results[lead]["RMSE"] += m_rmse
    results[lead]["R2"] += m_r2
    return results

if __name__== "__main__": 
    parser = argparse.ArgumentParser(description='castle')    
    parser.add_argument('-gpu', metavar='int', required=True, help='gpu index, using which GPU')
    parser.add_argument('-look', metavar='int', required=True, help='# days look forward')
    parser.add_argument('-lead', metavar='int', required=True, help='# days lead time')    
    args = parser.parse_args() 
    look = int(args.look)
    lead = int(args.lead)
    #set gpu for running experiment
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    no_gpu = 1
    
    print ('start constructing samples.')    

    esf, dsf, ob,fo, y_ob_sf,y_fo_sf = gs.get_samples(look = look, lead = lead, normalization = True)    
    
    train_esf, train_dsf,train_ob,train_fo,train_y_ob_sf,train_y_fo_sf = esf[:train_test], dsf[:train_test], ob[:train_test],fo[:train_test], y_ob_sf[:train_test],y_fo_sf[:train_test]
    
    test_esf, test_dsf,test_ob,test_fo,test_y_ob_sf,test_y_fo_sf = esf[train_test:], dsf[train_test:], ob[train_test:],fo[train_test:], y_ob_sf[train_test:],y_fo_sf[train_test:]
    
    print("training model...")
    clf = castle.CASTLE(batch_size, nb_epoch,observed_conf=(look, input_observed.shape[-1]), forecasted_conf = (lead,input_forecasted.shape[-1]),latent_dim = 256,batchNormalization=False, regularization=False)
    
    clf.fit([train_esf, train_dsf,train_ob,train_fo], [train_y_ob_sf,train_y_fo_sf])
    
    
    prediction = clf.predict(test_esf, test_ob,test_fo)
    
    results = {}
    for lead in ["5 days","7 days","10 days","15 days"]:
        results[lead] = {}
        for metric in ["MAE", "RMSE", "R2"]:
            results[model][metric] = 0
    
    
    
    results = get_metrics(results,"5 days", test_dsf[:,5], prediction[:,5])
    results = get_metrics(results,"7 days", test_dsf[:,7], prediction[:,7])
    results = get_metrics(results,"10 days", test_dsf[:,10], prediction[:,10])
    results = get_metrics(results,"15 days", test_dsf[:,15], prediction[:,15])
    df_results = pd.DataFrame(results)
    df_results.to_csv("castle_result.csv.gz.", index=None, header=True, compression="gzip")  
