import numpy as np
import sys
import argparse
import operator
import pandas as pd
#from __future__ import print_function
import os
import cPickle as pickle
import math
from time import time
from datetime import datetime
from datetime import timedelta
from netCDF4 import Dataset
#from keras.utils import multi_gpu_model


np.random.seed(1337)  # for reproducibility
nb_epoch = 1000  # number of epoch at training stage
batch_size = 15  # batch size
year_test = 2007 #2005
nb_flow = 1 # the number of channels 
start_year = 1985
end_year = 2005 #2016
lat0 = 17
lat1 = 32+8
lon0 = 70-8
lon1 = 101+8
lr=0
path_result = 'results/result'
path_model = 'results/model'
path_pic = 'results/pic'
path_log = 'results/log'
path_data = '/notebooks/workspace/Climate_Analysis/Data/'
    
def get_samples(target_river,lead,look): #G,B,M
    if target_river == 'G':
        target = pd.read_csv(path_data+'Ganges.csv',index_col=3,header=0,parse_dates=True)
    elif target_river == 'B':
        target = pd.read_csv(path_data+'Brahmaputra.csv',index_col=3,header=0,parse_dates=True)
    else:
        target = pd.read_csv(path_data+'Meghna.csv',index_col=3,header=0,parse_dates=True)
    idx_t = (target.index.year>(start_year-1)) & (target.index.year<(end_year+1)) & (target.index.month>5) & (target.index.month<10)
    y = target[idx_t]['Q (m3/s)']
    l = y.shape[0]
        
    persiann= Dataset(path_data+'PERSIANN_1_X_1_withlatlong.nc','r') #(12784, 117, 360) (1-12)
    observed=persiann.variables['precipitation'][:]
    timestamps = persiann.variables['time'][:] 
    timestamps= pd.to_datetime(timestamps,format='%Y-%m-%d') 
    dates=pd.DataFrame(timestamps, index=timestamps,dtype='datetime64[ns]')
    idx_o = (dates.index.year>(start_year-1)) & (dates.index.year<(end_year+1)) & (dates.index.month>5) & (dates.index.month<10)
    lat_o = persiann.variables['lat'][:] 
    lon_o = persiann.variables['lon'][:] 
    lat_idx = np.where(np.logical_and(lat_o>=lat0, lat_o<=lat1))
    lat_idx = lat_idx[0]
    lon_idx = np.where(np.logical_and(lon_o>=lon0, lon_o<=lon1))
    lon_idx = lon_idx[0]
    #print (observed.shape,lat_idx.shape,lon_idx.shape)
    observed = observed[:,lat_idx,:][:,:,lon_idx]
     
        
        
    stream = np.empty((l,0), float)
    ob = np.zeros((l,0,observed.shape[1],observed.shape[2]), float)
    i = 0
    for lag in np.arange(lead+look-1,lead-1,-1):
        target2 = target[:-lag]['Q (m3/s)']
        target2 = target2[:,np.newaxis]
        stream = np.append(stream, target2[idx_t[lag:]], axis=1)         
        ob2 = observed[:-lag]
        ob2=np.ma.fix_invalid(ob2)
        #print (ob.shape,' and ', ob2[idx_o[lag:]].shape)
        value = ob2[idx_o[lag:]]
        ob = np.append(ob, value[:,np.newaxis,...], axis=1)    
    #ob = np.transpose(ob, (0, 3, 1,2))  
    
    forecast = Dataset(path_data+'total_forecast_precipitation_mean_spread_corrected.nc','r')#(5049, 181, 360, 15, 2) (5,6,7,8,9)
    forecast_ms=forecast.variables['precipitation'][:]
    forecast_ms=np.ma.fix_invalid(forecast_ms)
    timestamps = forecast.variables['time'][:] 
    timestamps= pd.to_datetime(timestamps,format='%Y-%m-%d') 
    dates=pd.DataFrame(timestamps, index=timestamps,dtype='datetime64[ns]')
    idx_f = (dates.index.year>(start_year-1)) & (dates.index.year<(end_year+1)) & (dates.index.month>5) & (dates.index.month<10)
    print (dates[idx_f])
    lat_f = forecast.variables['lat'][:] 
    lon_f = forecast.variables['lon'][:] 
    lat_idx = np.where(np.logical_and(lat_f>=lat0, lat_f<=lat1))
    lat_idx = lat_idx[0]
    lon_idx = np.where(np.logical_and(lon_f>=lon0, lon_f<=lon1))
    lon_idx = lon_idx[0]
    forecast_ms = forecast_ms[:,lat_idx,:,:,:][:,:,lon_idx,:,:]
    f_ms2 = forecast_ms[:-lead+1]
    print (f_ms2.shape)
    fms = f_ms2[idx_f[lead-1:]][:,:,:,:lead,:]
    print (fms.shape)
    fms = np.transpose(fms, (0, 3, 1,2,4))    
    return stream,ob,fms,y # stream flow, Observation, Forecasts, Y
if __name__== "__main__": 
    stream,ob,fms,y = get_samples('G',3,3)
    print (type(stream),type(ob),type(fms),type(y))
    print (stream.shape,ob.shape,fms.shape,y.shape)
     
    np.save('stream',stream)    
    np.save('ob',ob)    
    np.save('fms',fms)    
    np.save('y',y)