import numpy as np
import scipy
import scipy.stats
import sys
import argparse
import operator
import pandas as pd
#from __future__ import print_function
import os
import cPickle as pickle
import math
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from datetime import timedelta
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard,LearningRateScheduler,ReduceLROnPlateau
from keras.models import model_from_json
from keras import backend as K
from sklearn import metrics
#from ActRecTrainingUtils import ActRecTrainingUtils, MomentumScheduler
import residualnet as rn
import conv_lstm_net as cln
import calibrating_net as unet
import densenet as dnet
import densenet3d as dnet3d
import resnet as rnet
import resnet3d as rnet3d
import nasnet
import crnn
import convLSTM
#from keras.utils import multi_gpu_model

#np.random.seed(1337)  # for reproducibility
np.random.seed(33) # NumPy
import random
#sta2= random.getstate()
#print('sta2 is: '+str(sta2))
random.seed(2) # Python
from tensorflow import set_random_seed, get_seed
#sta3 = get_seed
#print('sta3 is: '+str(sta3))
set_random_seed(3)





nb_epoch = 1000  # number of epoch at training stage
batch_size = 30  # batch size
test_train = 3060
train_validate = 3825
lr=0
path_result = 'results/result'
path_model = 'results/model'
path_pic = 'results/pic'
path_log = 'results/log'
path_data = '../Data/'

parser = argparse.ArgumentParser(description='sp_net')
parser.add_argument('-gpu', metavar='int', type=str, default='0', required=True, help='gpu index, using which GPU')
parser.add_argument('-lr', metavar = 'float', type=float, default=0.001, help='learning rate')
parser.add_argument('-lrtl', metavar='int', type=int, default=24, help='learning_rate_tail_len')
parser.add_argument('-model', metavar='str',type=str,default='unet2d', required=True, help='unet2d,unet3d,densenet121 169 201 264 161, resnet4 18 34 50 101 152')
parser.add_argument('-lead', metavar='int',type=int,default= 10, help='lead time')
parser.add_argument('-look', metavar='int',type=int,default= 2,  help='look up forward')
parser.add_argument('-norm', metavar='int',type=int,default= 0,  help='normalization, 0:non_normalization; 1:MinMax')
parser.add_argument('-kernel', metavar='str',type=str,default='s', help='kernel_size: s=(3*3); m=(5*5); l=(7*7)')
class PredictionCallback(Callback):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        m_mae0,m_rmse0,m_r20 = get_metrics(np.array(self.Y[0]).flatten(),np.array(pred[0]).flatten())
        m_mae1,m_rmse1,m_r21 = get_metrics(np.array(self.Y[1]).flatten(),np.array(pred[1]).flatten())
        m_mae2,m_rmse2,m_r22 = get_metrics(np.array(self.Y[2]).flatten(),np.array(pred[2]).flatten())
        print('Ganges Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae0, m_rmse0,m_r20))
        print('Brahmaputra Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae1, m_rmse1,m_r21))
        print('Meghna Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae2, m_rmse2,m_r22))

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''
    def __init__(self):
        pass
    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        #print("min:", self._min, "max:", self._max)
    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
def Normalization(Pob,Px_mean,Px_spread):
    #Yo= Y
    #Y = np.log(Y)     
    Pob_tr = Pob[0:train_validate]
    mmn = MinMaxNormalization()
    mmn.fit(Pob_tr)
    Pob = mmn.transform(Pob)     
       
    Px_mean_tr = Px_mean[0:train_validate]
    mmn = MinMaxNormalization()
    mmn.fit(Px_mean_tr)
    Px_mean = mmn.transform(Px_mean)
    
    Px_spread_tr = Px_spread[0:train_validate]
    mmn = MinMaxNormalization()
    mmn.fit(Px_spread_tr)
    Px_spread = mmn.transform(Px_spread)
    return Pob,Px_mean,Px_spread
def get_metrics(y, pred):
    out = pd.DataFrame(data={'y':y,'pred':pred})
    print(out.T)
    m_mae = metrics.mean_absolute_error(y, pred)
    m_rmse = metrics.mean_squared_error(y, pred)** 0.5
    m_r2 = metrics.r2_score(y, pred) 
    return m_mae,m_rmse,m_r2
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def compile_model(model):    
    #optimizer = SGD(lr=lr_schedule_fn(0), momentum=mom_schedule_fn(0))
    #optimizer = Adam(lr = 0.0001)
    #loss_weights = [1.0,1.5,2.0]
    model.compile(loss='mse', optimizer= 'adam', metrics=['mae'])
    #model.compile(loss='mse', optimizer= optimizer, metrics=['mae',r_squared])
    return model    
def build_model(model,o_conf, f_conf,nlags,kernel_size):    
    net=None
    if model.lower() == 'unet3d':#
        net = unet.net(f_conf,external_dim,kernel_size,is_3D=True) 
    elif model.lower() == 'unet2d':#
        net = unet.net(f_conf,external_dim,kernel_size,is_3D=False) 
    elif model.lower() == 'cnn':#
        net = rn.stresnet(f_conf, external_dim, kernel_size,is_3D=False,nb_conv = 5)
    elif model.lower() == 'densenet121_mt':#121 169 201 264 161
        net = dnet.net_mt(o_conf, f_conf, external_dim,121)
    elif model.lower() == 'densenet201_mt':#121 169 201 264 161
        net = dnet.net_mt(o_conf, f_conf, external_dim,201)
    elif model.lower() == 'densenet121':#121 169 201 264 161
        net = dnet.net(o_conf, f_conf, external_dim,121)
    elif model.lower() == 'densenet3d121_mt':#121 169 201 264 161
        net = dnet3d.net_mt(o_conf, f_conf, external_dim,121)
    elif model.lower() == 'densenet3d121_mt_v1':#121 169 201 264 161
        net = dnet3d.net_mt_v1(o_conf, f_conf, external_dim,121)
    elif model.lower() == 'densenet169':#
        net = dnet.net(o_conf, f_conf, external_dim,169)
    elif model.lower() == 'densenet201':#
        net = dnet.net(o_conf, f_conf, external_dim,201)
    elif model.lower() == 'densenet3d201_mt':
        net = dnet3d.net_mt(o_conf, f_conf, external_dim,201)
    elif model.lower() == 'densenet264':#
        net = dnet.net(o_conf, f_conf, external_dim,264)
    elif model.lower() == 'densenet161':#
        net = dnet.net(o_conf, f_conf, external_dim,161)
    elif model.lower() == 'densenet3d201':#
        net = dnet3d.net(o_conf, f_conf, external_dim,201)
    elif model.lower() == 'resnet18_v2':
        net = rnet.net_v2(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet4':
        net = rnet.net(o_conf, f_conf, external_dim,4)
    elif model.lower() == 'resnet18':#4 18 34 50 101 152
        net = rnet.net(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet18_mt':
        net = rnet.net_mt(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet18_mt_v1':
        net = rnet.net_mt_v1(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet34_mt':
        net = rnet.net_mt(o_conf, f_conf, external_dim,34)
    elif model.lower() == 'resnet50_mt':
        net = rnet.net_mt(o_conf, f_conf, external_dim,50)
    elif model.lower() == 'resnet101_mt':
        net = rnet.net_mt(o_conf, f_conf, external_dim,101)
    elif model.lower() == 'resnet152_mt':
        net = rnet.net_mt(o_conf, f_conf, external_dim,152)
    elif model.lower() == 'resnet34':
        net = rnet.net(o_conf, f_conf, external_dim,34)
    elif model.lower() == 'resnet50':
        net = rnet.net(o_conf, f_conf, external_dim,50)
    elif model.lower() == 'resnet101':
        net = rnet.net(o_conf, f_conf, external_dim,101)
    elif model.lower() == 'resnet152':
        net = rnet.net(o_conf, f_conf, external_dim,152)
    elif model.lower() == 'resnet3d18':
        net = rnet3d.net(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_mt':
        net = rnet3d.net_mt(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_mt_1':
        net = rnet3d.net_mt_1(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_mt4':
        net = rnet3d.net_mt4(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_mt5':
        net = rnet3d.net_mt5(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_mt_v1':
        net = rnet3d.net_mt_v1(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'resnet3d18_v2':
        net = rnet3d.net_v2(o_conf, f_conf, external_dim,18)
    elif model.lower() == 'nasnet_large':#large mobile cifar
        net = nasnet.net(o_conf, f_conf, external_dim,'large')
    elif model.lower() == 'nasnet_mobile':
        net = nasnet.net(o_conf, f_conf, external_dim,'mobile')
    elif model.lower() == 'nasnet_cifar':
        net = nasnet.net(o_conf, f_conf, external_dim,'cifar')
    elif model.lower() == 'crnn3d16':
        net = crnn.net(o_conf, f_conf, external_dim,'16')
    elif model.lower() == 'crnn3d16_v2':
        net = crnn.net_v2(o_conf, f_conf, external_dim,'16')
    elif model.lower() == 'convlstm3d':
        net = convLSTM.net(o_conf, f_conf, external_dim,'1')
    net= compile_model(net)
    #net.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True) 
    return net
def get_samples(lead,look,is3d,normalization):
    lead1 = 10
    look1 = 15
    Qx_Ganges = pd.read_csv(path_data+'X_Ganges.csv')
    idx = []
    for i in np.arange(1-look1,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Ganges = np.array(Qx_Ganges.loc[:,idx])
    Qy_Ganges = pd.read_csv(path_data+'Y_Ganges.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Ganges = np.array(Qy_Ganges.loc[:,idy])
    
    
    Qx_Brahmaputra = pd.read_csv(path_data+'X_Brahmaputra.csv')
    idx = []
    for i in np.arange(1-look1,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Brahmaputra = np.array(Qx_Brahmaputra.loc[:,idx])
    Qy_Brahmaputra = pd.read_csv(path_data+'Y_Brahmaputra.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Brahmaputra = np.array(Qy_Brahmaputra.loc[:,idy])
    
    
    Qx_Meghna = pd.read_csv(path_data+'X_Meghna.csv')
    idx = []
    for i in np.arange(1-look1,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Meghna = np.array(Qx_Meghna.loc[:,idx])
    Qy_Meghna = pd.read_csv(path_data+'Y_Meghna.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Meghna = np.array(Qy_Meghna.loc[:,idy])
    
    
    
    
#     inx = np.load(path_data+'total_forecast_precipitation_mean_spread_input.npy')
#     Px_mean = inx[:,:,:,:lead1,:][:,:,:,:,0]    
#     Px_spread = inx[:,:,:,:lead1,:][:,:,:,:,1]  
    ino = np.load(path_data+'persiann_1_x_1_look20.npy')
    Pob = ino[:,-look1:,:,:]#(4896, 15, 64, 128)
    time,external_dim=Qx_Ganges.shape
    samples, timesteps, map_height, map_width  = Pob.shape 
    channels = 1
#    if is3d==0:
#         Px_mean = np.moveaxis(Px_mean,3,1) 
#         samples, timesteps, map_height, map_width = Px_mean.shape 
#         Px_spread = np.moveaxis(Px_spread,3,1) 
#    else:   
#         Px_mean = Px_mean[:,np.newaxis,:,:,:]
#         Px_mean = np.moveaxis(Px_mean,4,2)  
    Pob = Pob[:,np.newaxis,:,:,:]
#         samples, __,timesteps, map_height, map_width = Px_mean.shape 
#         Px_spread = Px_spread[:,np.newaxis,:,:,:]
#         Px_spread = np.moveaxis(Px_spread,4,2)
    if normalization:
        print ('MinMax normalization')
        #Pob,Px_mean,Px_spread = Normalization (Pob,Px_mean,Px_spread)  
    #if is3d==2:
        #Px = np.append(Px_mean,Px_spread,axis=1)
        #channels = 2
        #X_tr = [Pob[0:train_validate],Px[0:train_validate],Qx[0:train_validate]] 
        #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
        #X_te = [Pob[train_validate:],Px[train_validate:],Qx[train_validate:]] 
    
    #else:
    Pfo = Pob[lead:,:,:,:,:][:,:,-lead:,:,:]
    print("pfo shape: "+str(Pfo.shape))
    Pob = Pob[:-lead,:,:,:,:]
    print("pob shape: "+str(Pob.shape))
    Qx_Ganges = Qx_Ganges[:-lead]
    Qy_Ganges = Qy_Ganges[:-lead]
    Qx_Brahmaputra = Qx_Brahmaputra[:-lead]
    Qy_Brahmaputra = Qy_Brahmaputra[:-lead]
    Qx_Meghna = Qx_Meghna[:-lead]
    Qy_Meghna = Qy_Meghna[:-lead]
   
    X_tr = [Pob[0:train_validate],Pfo[0:train_validate],Qx_Ganges[0:train_validate],Qx_Brahmaputra[0:train_validate],Qx_Meghna[0:train_validate]] 
    #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
    X_te = [Pob[train_validate:],Pfo[train_validate:],Qx_Ganges[train_validate:],Qx_Brahmaputra[train_validate:],Qx_Meghna[train_validate:]]
    
    #X_tr = [Px_spread[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_spread[train_validate:],Qx[train_validate:]] 
    
    

    #X_tr = [Px_mean[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Qx[train_validate:]] 
    
    
    #X_tr = [Px_mean[0:test_train],Px_spread[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Px_spread[train_validate:]]    
    Y_tr =  [Qy_Ganges[0:train_validate],Qy_Brahmaputra[0:train_validate],Qy_Meghna[0:train_validate]]      
    Y_te =  [Qy_Ganges[train_validate:],Qy_Brahmaputra[train_validate:],Qy_Meghna[train_validate:]]   
     
    f_conf = (lead1, channels, map_height, map_width) if lead > 0 else None
    o_conf = (look1, 1, map_height, map_width) if look > 0 else None
    return X_tr,X_te,Y_tr,Y_te,o_conf, f_conf,external_dim
def get_samples_1(lead,look,is3d,normalization):    
    look = 20
    lead = 10
    Qx_Ganges = pd.read_csv(path_data+'X_Ganges.csv')
    idx = []
    for i in np.arange(1-look,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Ganges = np.array(Qx_Ganges.loc[:,idx])
    Qy_Ganges = pd.read_csv(path_data+'Y_Ganges.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Ganges = np.array(Qy_Ganges.loc[:,idy])
    
    
    Qx_Brahmaputra = pd.read_csv(path_data+'X_Brahmaputra.csv')
    idx = []
    for i in np.arange(1-look,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Brahmaputra = np.array(Qx_Brahmaputra.loc[:,idx])
    Qy_Brahmaputra = pd.read_csv(path_data+'Y_Brahmaputra.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Brahmaputra = np.array(Qy_Brahmaputra.loc[:,idy])
    
    
    Qx_Meghna = pd.read_csv(path_data+'X_Meghna.csv')
    idx = []
    for i in np.arange(1-look,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx_Meghna = np.array(Qx_Meghna.loc[:,idx])
    Qy_Meghna = pd.read_csv(path_data+'Y_Meghna.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy_Meghna = np.array(Qy_Meghna.loc[:,idy])
    
    inx = np.load(path_data+'total_forecast_precipitation_mean_spread_input.npy')
    Px_mean = inx[:,:,:,:lead,:][:,:,:,:,0]    
    Px_spread = inx[:,:,:,:lead,:][:,:,:,:,1]  
    ino = np.load(path_data+'persiann_1_x_1_look20.npy')
    Pob = ino[:,-look:,:,:]#(4896, 15, 64, 128)
    time,external_dim=Qx_Ganges.shape
    samples, map_height, map_width, timesteps = Px_mean.shape 
    
    len_closeness = 5
    len_period = 3
    len_trend = 3
    PeriodInterval = 1
    TrendInterval = 1
    T0 = 3
    T1 = 4
    T2 = 6
    depends = [range(0, len_closeness,1),
               [PeriodInterval * T0 * j for j in range(1, len_period+1)],
               range(look-1, look-len_closeness-1,-1),
               [look-PeriodInterval * T1 * j for j in range(1, len_period+1)],
               [look-TrendInterval * T2 * j for j in range(1, len_trend+1)]]
    print depends
   
    
    channels = 1

    if is3d==0:
        Px_mean = np.moveaxis(Px_mean,3,1) 
        samples, timesteps, map_height, map_width = Px_mean.shape 
        Px_spread = np.moveaxis(Px_spread,3,1) 
    else:   
        Px_mean = Px_mean[:,np.newaxis,:,:,:]
        Px_mean = np.moveaxis(Px_mean,4,2)  
        Pob = Pob[:,np.newaxis,:,:,:]
        samples, __,timesteps, map_height, map_width = Px_mean.shape 
        Px_spread = Px_spread[:,np.newaxis,:,:,:]
        Px_spread = np.moveaxis(Px_spread,4,2)

    if normalization:
        print ('MinMax normalization')
        Pob,Px_mean,Px_spread = Normalization (Pob,Px_mean,Px_spread)  
    #if is3d==2:
        #Px = np.append(Px_mean,Px_spread,axis=1)
        #channels = 2
        #X_tr = [Pob[0:train_validate],Px[0:train_validate],Qx[0:train_validate]] 
        #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
        #X_te = [Pob[train_validate:],Px[train_validate:],Qx[train_validate:]] 
    
    #else:
    if is3d==0:
        X_tr = [Pob[0:train_validate][:,depends[4]],Pob[0:train_validate][:,depends[3]],Pob[0:train_validate][:,depends[2]],Px_mean[0:train_validate][:,depends[1]],Px_mean[0:train_validate][:,depends[0]],Px_spread[0:train_validate][:,depends[1]],Px_spread[0:train_validate][:,depends[0]],Qx_Ganges[0:train_validate],Qx_Brahmaputra[0:train_validate],Qx_Meghna[0:train_validate]] 
        #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
        X_te = [Pob[train_validate:][:,depends[4]],Pob[train_validate:][:,depends[3]],Pob[train_validate:][:,depends[2]],Px_mean[train_validate:][:,depends[1]],Px_mean[train_validate:][:,depends[0]],Px_spread[train_validate:][:,depends[1]],Px_spread[train_validate:][:,depends[0]],Qx_Ganges[train_validate:],Qx_Brahmaputra[train_validate:],Qx_Meghna[train_validate:]]

    else:   
        X_tr = [Pob[0:train_validate][:,:,depends[4]],Pob[0:train_validate][:,:,depends[3]],Pob[0:train_validate][:,:,depends[2]],Px_mean[0:train_validate][:,:,depends[1]],Px_mean[0:train_validate][:,:,depends[0]],Px_spread[0:train_validate][:,:,depends[1]],Px_spread[0:train_validate][:,:,depends[0]],Qx_Ganges[0:train_validate],Qx_Brahmaputra[0:train_validate],Qx_Meghna[0:train_validate]] 
        #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
        X_te = [Pob[train_validate:][:,:,depends[4]],Pob[train_validate:][:,:,depends[3]],Pob[train_validate:][:,:,depends[2]],Px_mean[train_validate:][:,:,depends[1]],Px_mean[train_validate:][:,:,depends[0]],Px_spread[train_validate:][:,:,depends[1]],Px_spread[train_validate:][:,:,depends[0]],Qx_Ganges[train_validate:],Qx_Brahmaputra[train_validate:],Qx_Meghna[train_validate:]]
    #X_tr = [Px_spread[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_spread[train_validate:],Qx[train_validate:]] 
    
    

    #X_tr = [Px_mean[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Qx[train_validate:]] 
    
    
    #X_tr = [Px_mean[0:test_train],Px_spread[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Px_spread[train_validate:]]    
    Y_tr =  [Qy_Ganges[0:train_validate],Qy_Brahmaputra[0:train_validate],Qy_Meghna[0:train_validate]]      
    Y_te =  [Qy_Ganges[train_validate:],Qy_Brahmaputra[train_validate:],Qy_Meghna[train_validate:]]  
    o_conf = {}   
    f_conf = {} 
    o_conf[0] = (len_trend, 1, map_height, map_width) if look > 0 else None
    o_conf[1] = (len_period, 1, map_height, map_width) if look > 0 else None
    o_conf[2] = (len_closeness, 1, map_height, map_width) if look > 0 else None
    f_conf[0] = (len_period, channels, map_height, map_width) if lead > 0 else None
    f_conf[1] = (len_closeness, channels, map_height, map_width) if lead > 0 else None
    return X_tr,X_te,Y_tr,Y_te,o_conf, f_conf,external_dim
def init_fun(args):  
    lr = args.lr
    if "3d" in args.model:
        if "_v2" in args.model:
            args.is3d = 2
        else:
            args.is3d = 1
        if args.kernel=='s':
            args.kernel=(2, 3, 3)
        elif args.kernel=='m':
            args.kernel=(2, 5, 5)
        elif args.kernel=='l':
            args.kernel=(2, 7, 7)  
    else:
        args.is3d = 0
        if args.kernel=='s':
            args.kernel=(3, 3)
        elif args.kernel=='m':
            args.kernel=(5, 5)
        elif args.kernel=='l':
            args.kernel=(7, 7)  
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    if os.path.isdir(path_model) is False:
        os.mkdir(path_model)
    if os.path.isdir(path_pic) is False:
        os.mkdir(path_pic)
    if os.path.isdir(path_log) is False:
        os.mkdir(path_log)
if __name__== "__main__": 
    args = parser.parse_args()
    init_fun(args)
    

    #set gpu for running experiment
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    no_gpu = 1
    
    print ('start constructing samples.') 
    if args.model == 'resnet3d18_mt4':
        X_tr,X_te,Y_tr,Y_te,o_conf, f_conf,external_dim = get_samples_1(args.lead,args.look,args.is3d,args.norm)
    else:
        X_tr,X_te,Y_tr,Y_te,o_conf, f_conf,external_dim = get_samples(args.lead,args.look,args.is3d,args.norm)
    print ('X_tr shape', X_tr[0].shape,'X_te shape', X_te[0].shape) 
    
    
    
    print ('start building model.') 
    model = build_model(args.model,o_conf, f_conf,external_dim,args.kernel)
    hyperparams_name = 'test_dms.{}.{}.{}.{}.{}'.format(args.model,args.lead,args.look,args.kernel,args.lr)
    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4)
    model_checkpoint = ModelCheckpoint(fname_param, verbose=1, save_best_only=True, mode='min',save_weights_only=True) 

    print("training model...") 
    predcall = PredictionCallback(X_te,Y_te)
    predcall.set_model(model)  
    path_log = os.path.join(path_log, '{}'.format(hyperparams_name))
    tb = TensorBoard(log_dir=path_log, histogram_freq=5, batch_size=batch_size, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)#, update_freq='epoch'
    #history = model.fit(X_tr, Y_train,epochs=nb_epoch,batch_size=batch_size,validation_split=0.1,
                        #callbacks=[model_checkpoint, predcall,tb],verbose=1)#early_stopping, 
    #lr_scheduler = LearningRateScheduler(lr_schedule_fn, verbose=1)
    #mom_scheduler = MomentumScheduler(mom_schedule_fn, verbose=1)

    callbacks = [model_checkpoint, early_stopping,reduce_lr,predcall]#,tb
    
    
    history = model.fit(X_tr, Y_tr,
                epochs=nb_epoch,
                batch_size=batch_size,
                #validation_split=0.1,
                validation_data=(X_te,Y_te),
                callbacks=callbacks,
                verbose=1)
    #history = model.fit(X_tr, Y_train, epochs=nb_epoch, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint, early_stopping,predcall,tb], validation_data=(X_val,Y_val))#early_stopping,
    
    model_json = model.to_json()
    with open(os.path.join(
        path_model, '{}.json'.format(hyperparams_name)), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print('=' * 10)
    
    
    
    print('evaluating using the model that has the best loss on the valid set')
    json_file = open(os.path.join(
        path_model, '{}.json'.format(hyperparams_name)), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)    
    model.load_weights(fname_param)
    model = compile_model(model)
    prediction = model.predict(X_tr,verbose=0)
    m_mae0,m_rmse0,m_r20 = get_metrics(np.array(Y_tr[0]).flatten(),np.array(prediction[0]).flatten())
    m_mae1,m_rmse1,m_r21 = get_metrics(np.array(Y_tr[1]).flatten(),np.array(prediction[1]).flatten())
    m_mae2,m_rmse2,m_r22 = get_metrics(np.array(Y_tr[2]).flatten(),np.array(prediction[2]).flatten())
    print('Ganges Training score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae0, m_rmse0,m_r20))
    print('Brahmaputra Training score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae1, m_rmse1,m_r21))
    print('Meghna Training score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae2, m_rmse2,m_r22))
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.train_prediction.pkl'.format(hyperparams_name)), 'wb'))
    prediction = model.predict(X_te,verbose=0)
    m_mae0,m_rmse0,m_r20 = get_metrics(np.array(Y_te[0]).flatten(),np.array(prediction[0]).flatten())
    m_mae1,m_rmse1,m_r21 = get_metrics(np.array(Y_te[1]).flatten(),np.array(prediction[1]).flatten())
    m_mae2,m_rmse2,m_r22 = get_metrics(np.array(Y_te[2]).flatten(),np.array(prediction[2]).flatten())
    print('Ganges Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae0, m_rmse0,m_r20))
    print('Brahmaputra Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae1, m_rmse1,m_r21))
    print('Meghna Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae2, m_rmse2,m_r22))
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.test_prediction.pkl'.format(hyperparams_name)), 'wb'))
    
    
    
