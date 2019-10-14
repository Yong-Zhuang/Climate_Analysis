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
import resnet as rnet
#from keras.utils import multi_gpu_model

np.random.seed(1337)  # for reproducibility
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
parser.add_argument('-model', metavar='str',type=str,default='unet2d', required=True, help='unet2d,unet3d,densenet121 169 201 264 161, resnet18 34 50 101 152')
parser.add_argument('-lead', metavar='int',type=int,default= 10, help='lead time')
parser.add_argument('-look', metavar='int',type=int,default= 2,  help='look up forward')
parser.add_argument('-kernel', metavar='str',type=str,default='s', help='kernel_size: s=(3*3); m=(5*5); l=(7*7)')
class PredictionCallback(Callback):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        m_mae,m_rmse,m_r2 = get_metrics(np.array(self.Y).flatten(),np.array(pred).flatten())
        print('Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae, m_rmse,m_r2))

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
def Normalization(X,Y,E,train_idx,mode):
    #Yo= Y
    #Y = np.log(Y)
    if mode ==3:
        return X,Y,E
    if mode==1:
        X =  np.log(1+X)
        E =  np.log(E)        
    data_train = X[train_idx]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    X = mmn.transform(X)     
       
    E_tr = E[train_idx]
    mmn = MinMaxNormalization()
    mmn.fit(E_tr)
    E = mmn.transform(E)
    return X,Y,E
def get_metrics(y, pred):
    idx = np.any(np.isnan(pred))
    pred[idx]=0
    idx = np.any(np.isfinite(pred)==False)
    pred[idx]=0
    idx = pred>12
    print('number of >12',sum(idx))
    pred[idx]=12
    pre = np.exp(pred)
    
    out = pd.DataFrame(data={'y':y,'pred':pre})
    print(out.T)
    m_mae = metrics.mean_absolute_error(y, pre)
    m_rmse = metrics.mean_squared_error(y, pre)** 0.5
    m_r2 = metrics.r2_score(y, pre) 
    return m_mae,m_rmse,m_r2
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def compile_model(model):    
    #optimizer = SGD(lr=lr_schedule_fn(0), momentum=mom_schedule_fn(0))
    optimizer = Adam(lr = 0.0001)
    #model.compile(loss='mse', optimizer= 'adam', metrics=['mae',r_squared])
    model.compile(loss='mse', optimizer= optimizer, metrics=['mae',r_squared])
    return model    
def build_model(model,conf,nlags,kernel_size):    
    net=None
    if model.lower() == 'unet3d':#
        net = unet.net(conf,external_dim,kernel_size,is_3D=True) 
    elif model.lower() == 'unet2d':#
        net = unet.net(conf,external_dim,kernel_size,is_3D=False) 
    elif model.lower() == 'cnn':#
        net = rn.stresnet(conf, external_dim, kernel_size,is_3D=False,nb_conv = 5)
    elif model.lower() == 'densenet121':#121 169 201 264 161
        net = dnet.net(conf, external_dim,121)
    elif model.lower() == 'densenet169':#
        net = dnet.net(conf, external_dim,169)
    elif model.lower() == 'densenet201':#
        net = dnet.net(conf, external_dim,201)
    elif model.lower() == 'densenet264':#
        net = dnet.net(conf, external_dim,264)
    elif model.lower() == 'densenet161':#
        net = dnet.net(conf, external_dim,161)
    elif model.lower() == 'resnet18':#18 34 50 101 152
        net = rnet.net(conf, external_dim,18)
    elif model.lower() == 'resnet34':
        net = rnet.net(conf, external_dim,34)
    elif model.lower() == 'resnet50':
        net = rnet.net(conf, external_dim,50)
    elif model.lower() == 'resnet101':
        net = rnet.net(conf, external_dim,101)
    elif model.lower() == 'resnet152':
        net = rnet.net(conf, external_dim,152)
    net= compile_model(net)
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return net
def get_samples(lead,look,is3d):
    Qx = pd.read_csv(path_data+'X_Ganges.csv')
    idx = []
    for i in np.arange(1-look,1,1):
        idx.append('Q_'+str(i))
    print (idx)
    Qx = np.array(Qx.loc[:,idx])
    Qy = pd.read_csv(path_data+'Y_Ganges.csv')
    idy = 'Q_'+str(lead)
    print (idy)
    Qy = np.array(Qy.loc[:,idy])
    inx = np.load(path_data+'total_forecast_precipitation_mean_spread_input.npy')
    Px_mean = inx[:,:,:,:lead,:][:,:,:,:,0]    
    Px_spread = inx[:,:,:,:lead,:][:,:,:,:,1]  
    time,external_dim=Qx.shape
    samples, map_height, map_width, timesteps = Px_mean.shape 
    if is3d==1:   
        Px_mean = Px_mean[:,np.newaxis,:,:,:]
        Px_mean = np.moveaxis(Px_mean,4,2)  
        samples, __,timesteps, map_height, map_width = Px_mean.shape 
        Px_spread = Px_spread[:,np.newaxis,:,:,:]
        Px_spread = np.moveaxis(Px_spread,4,2)
    else:
        Px_mean = np.moveaxis(Px_mean,3,1) 
        samples, timesteps, map_height, map_width = Px_mean.shape 
        Px_spread = np.moveaxis(Px_spread,3,1) 

    X_tr = [Px_mean[0:train_validate],Px_spread[0:train_validate],Qx[0:train_validate]] 
    #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
    X_te = [Px_mean[train_validate:],Px_spread[train_validate:],Qx[train_validate:]] 


    #X_tr = [Px_spread[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_spread[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_spread[train_validate:],Qx[train_validate:]] 
    
    

    #X_tr = [Px_mean[0:test_train],Qx[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Qx[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Qx[train_validate:]] 
    
    
    #X_tr = [Px_mean[0:test_train],Px_spread[0:test_train]] 
    #X_val = [Px_mean[test_train:train_validate],Px_spread[test_train:train_validate]] 
    #X_te = [Px_mean[train_validate:],Px_spread[train_validate:]] 
    Y = np.log(Qy)
    log_Y_tr = Y[0:train_validate]
    #Y_val =  Qy[test_train:train_validate]
    log_Y_te = Y[train_validate:]    
    Y_tr =  Qy[0:train_validate]      
    Y_te =  Qy[train_validate:]   
     
    conf = (timesteps, 1, map_height, map_width) if timesteps > 0 else None
    return X_tr,X_te,log_Y_tr,log_Y_te,Y_tr,Y_te, conf,external_dim
def init_fun(args):  
    lr = args.lr
    if args.model=='unet3d':
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
    X_tr,X_te,log_Y_tr,log_Y_te,Y_tr,Y_te,conf,external_dim = get_samples(args.lead,args.look,args.is3d)
    print ('X_tr shape', X_tr[0].shape,'X_te shape', X_te[0].shape) 
    
    
    
    print ('start building model.') 
    model = build_model(args.model,conf,external_dim,args.kernel)
    hyperparams_name = 'test_dms.{}.{}.{}.{}.{}'.format(args.model,args.lead,args.look,args.kernel,args.lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    model_checkpoint = ModelCheckpoint(fname_param, verbose=1, save_best_only=True, mode='min',save_weights_only=True) 

    print("training model...") 
    predcall = PredictionCallback(X_te,Y_te)
    predcall.set_model(model)  
    path_log = os.path.join(path_log, '{}'.format(hyperparams_name))
    tb = TensorBoard(log_dir=path_log, histogram_freq=5, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)#, update_freq='epoch'
    #history = model.fit(X_tr, Y_train,epochs=nb_epoch,batch_size=batch_size,validation_split=0.1,
                        #callbacks=[model_checkpoint, predcall,tb],verbose=1)#early_stopping, 
    #lr_scheduler = LearningRateScheduler(lr_schedule_fn, verbose=1)
    #mom_scheduler = MomentumScheduler(mom_schedule_fn, verbose=1)

    callbacks = [model_checkpoint, early_stopping,reduce_lr,predcall]#,tb
    
    
    history = model.fit(X_tr, log_Y_tr,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_data=(X_te,log_Y_te),
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
    m_mae,m_rmse,m_r2 = get_metrics(np.array(Y_tr).flatten(),np.array(prediction).flatten())
    print('Training score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae, m_rmse,m_r2))
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.train_prediction.pkl'.format(hyperparams_name)), 'wb'))
    prediction = model.predict(X_te,verbose=0)
    m_mae,m_rmse,m_r2 = get_metrics(np.array(Y_te).flatten(),np.array(prediction).flatten())
    print('Test score: %.6f rmse (norm): %.6f r_squared: %.6f' % (m_mae, m_rmse,m_r2))
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.test_prediction.pkl'.format(hyperparams_name)), 'wb'))
    
    
    
