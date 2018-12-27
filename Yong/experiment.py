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
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from keras.models import model_from_json
from keras import backend as K
import residualnet as rn
import conv_lstm_net as cln
#from keras.utils import multi_gpu_model


np.random.seed(1337)  # for reproducibility
nb_epoch = 1000  # number of epoch at training stage
batch_size = 15  # batch size
year_test = 2005
nb_flow = 1 # the number of channels 
start_year = 1985
end_year = 2016
lat0 = 17
lat1 = 32+8
lon0 = 70-8
lon1 = 101+8
lr=0
path_result = 'results/result'
path_model = 'results/model'
path_pic = 'results/pic'
path_log = 'results/log'
path_data = '../Data/'
class PredictionCallback(Callback):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X)
        out = pd.DataFrame(data={'y':np.array(self.Y).flatten(),'pred':np.array(pred).flatten()})
        print(out.T)

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

def get_streamflow(target_river,dayN,day0): #G,B,M
    if target_river == 'G':
        target = pd.read_csv(path_data+'Ganges.csv',index_col=3,header=0,parse_dates=True)
    elif target_river == 'B':
        target = pd.read_csv(path_data+'Brahmaputra.csv',index_col=3,header=0,parse_dates=True)
    else:
        target = pd.read_csv(path_data+'Meghna.csv',index_col=3,header=0,parse_dates=True)
    dates = (target.Year > (start_year-1)) & (target.Year<(end_year+1))
    target2 = target[dates]
    #print (target2)
    #print(target2[:50])
    dates = (target2.Month>5) & (target2.Month<10)
    target2 = target2.loc[dates]
    #print(target2.index[:50])
    #target2 = target2.reset_index()
    #target2.index = target2.index - timedelta(15)
    #print(target2.index[:50])
    #print(list(target2))
    frame = pd.DataFrame(target2['Q (m3/s)'])
    frame.columns = ['Q']
    for lag in np.arange(dayN,day0+1):
        x = target.loc[target2.index - timedelta(lag), 'Q (m3/s)' ]
        x = pd.DataFrame(x)
        x.columns = [''.join(['Q_',str(lag)])]   
        x.index = frame.index
        frame = pd.concat([frame,x],axis=1)
    frame.to_csv(path_data+'dframe.csv')
    return frame,target2
def get_precipitation(dataset, isForecast):
    if dataset=='5':
        precip = np.load(path_data+'precip5.npy') 
    else:
        precip = np.load(path_data+'precip10.npy') 
    #print (precip.shape)   
    timestamps = np.load(path_data+'time.npy')    
    timestamps= pd.to_datetime(timestamps,format='%Y-%m-%d') 
    dates=pd.DataFrame(timestamps, index=timestamps,dtype='datetime64[ns]')
    idx  = (dates.index.year>(start_year-1))&(dates.index.year<(end_year+1)) 
    lat = np.load(path_data+'lat.npy')
    lon = np.load(path_data+'lon.npy')
    lat_idx = np.where(np.logical_and(lat>=lat0, lat<=lat1))
    lat_idx = np.array(lat_idx).flatten()
    lon_idx = np.where(np.logical_and(lon>=lon0, lon<=lon1))
    lon_idx = np.array(lon_idx).flatten()
    precip = precip[idx,:,:,:][:,:,lat_idx,:][:,:,:,lon_idx]
    if isForecast=='1': #only use observision data
        if dataset=='5':
            precip= precip[:,0:15,:,:]
        else:
            precip = precip[:,0:10,:,:] 
    return precip

def Normalization(X,Y,E,train_idx,mode):
    Y = np.log(Y)
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
#Root Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5
#Root Mean Squared Logarithmic Error
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(tf.log1p(y_pred) - tf.log1p(y_true))))


def compile_model(model):
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])
    #model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model
    
def build_model(model,conf,external_dim,kernel,filters,no_layers):    
    net=None
    if model.lower() == 'cnn':#CNN 
        net = rn.stresnet(conf=conf, external_dim=external_dim, kernel_size=kernel, filters=filters, nb_resunit=0, nb_conv=no_layers, is_3D=False, batchNormalization=False) 
    elif model.lower() == 'resnet':#residual network
        net = rn.stresnet(conf=conf, external_dim=external_dim, kernel_size=kernel, filters=filters, nb_resunit=no_layers, nb_conv=0, is_3D=False, batchNormalization=False) 
    elif model.lower() == 'convlstm':#convolutional LSTM  
        net = cln.convLSTM_net(conf=conf, external_dim=external_dim, kernel_size=kernel, filters=filters, nb_stack=no_layers, batchNormalization=False) 
    elif model.lower() == 'convlstm_nr':#convolutional LSTM  
        net = cln.convLSTM_net(conf=conf, external_dim=external_dim, kernel_size=kernel, filters=filters, nb_stack=no_layers, batchNormalization=False,regularization=False) 
    net= compile_model(net)
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return net

if __name__== "__main__": 
    parser = argparse.ArgumentParser(description='sp_net')
    parser.add_argument('-data', metavar='int', required=True, help='10=10_day_forecast.nc; 5=5_day_forecast.nc')
    parser.add_argument('-forecast', metavar='int', required=True, help='0=using forecasting data; 1=no forecasting data')
    parser.add_argument('-gpu', metavar='int', required=True, help='gpu index, using which GPU')
    parser.add_argument('-lr', metavar = 'float', required=True, help='learning rate')
    parser.add_argument('-model', metavar='str', required=True, help='cnn, resnet, convlstm, convlstm_nr...')
    parser.add_argument('-layers', metavar='int', required=True, help='# of layers')
    parser.add_argument('-filters', metavar='int', required=True, help='# of filters in each layer')
    parser.add_argument('-kernel', metavar='str', required=True, help='kernel_size: s=(3*3); m=(5*5); l=(7*7)')
    parser.add_argument('-norm', metavar='int', required=True, help='normalization, 1: log+minmax; 2:minmax; 3:none')
    args = parser.parse_args() 
    lr = float(args.lr)
    args.layers = int(args.layers)
    args.filters = int(args.filters)
    if args.kernel=='s':
        args.kernel=(3, 3)
    elif args.kernel=='m':
        args.kernel=(5, 5)
    elif args.kernel=='l':
        args.kernel=(7, 7)        
    args.norm = int(args.norm)
    start= time()    
    
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    if os.path.isdir(path_model) is False:
        os.mkdir(path_model)
    if os.path.isdir(path_pic) is False:
        os.mkdir(path_pic)
    if os.path.isdir(path_log) is False:
        os.mkdir(path_log)
    #set gpu for running experiment
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    no_gpu = 1
    
    print ('start constructing samples.')    
    X=get_precipitation(args.data,args.forecast)  
    frame,target2 = get_streamflow('G',16,20) # get waterflow from day-20 to day-16 and day0, 6 days data.
    Y = frame.Q # waterflow at day0.
    frame.drop('Q',axis=1,inplace=True)
    E = frame.values # waterflow from day-20 to day-16, 5 days data.
    print('X shape', X.shape,'E shape', E.shape,'Y shape', Y.shape)    
    train_idx  = target2.Year < year_test
    test_idx  = target2.Year >= year_test
    print ('start normalizating')
    X,Y,E = Normalization(X,Y,E,train_idx,args.norm)
        
    E_train, X_train, Y_train, E_test, X_test, Y_test = E[train_idx], X[train_idx],Y[train_idx],E[test_idx],X[test_idx],Y[test_idx]     
    print ('X_train shape', X_train.shape, 'E_train shape', E_train.shape, 'Y_train shape', Y_train.shape,'X_test shape', X_test.shape, 'E_test shape', E_test.shape, 'Y_test shape', Y_test.shape)
    
    
    print ('start building model.') 
    samples, timesteps, map_height, map_width = X_train.shape 
    external_dim = E_train.shape[1] # water flow
    conf = (timesteps, 1, map_height,
              map_width) if timesteps > 0 else None
    model = build_model(args.model,conf,external_dim,args.kernel,args.filters,args.layers)
    hyperparams_name = 'd{}.c{}.f{}.{}.lr{}.k{}.f{}.l{}.norm{}'.format(
        args.data,conf,args.forecast,args.model,args.lr,args.kernel,args.filters,args.layers,args.norm)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')    

    print("training model...")
    if args.model.lower() == 'convlstm' or args.model.lower() == 'convlstm_nr':
        X_tr=[X_train[:,:, np.newaxis,...],E_train]
        X_te=[X_test[:,:, np.newaxis,...],E_test]
        X_pc = [X_te[0][:50],E_test[:50]]
    else:
        X_tr=[X_train,E_train]
        X_te=[X_test,E_test]
        X_pc = [X_test[:50],E_test[:50]]    
    print ('X_tr shape', X_tr[0].shape,'X_te shape', X_te[0].shape, 'X_pc shape', X_pc[0].shape)    
    
    predcall = PredictionCallback(X_pc,Y_test[:50])
    predcall.set_model(model)  
    path_log = os.path.join(path_log, '{}'.format(hyperparams_name))
    tb = TensorBoard(log_dir=path_log, histogram_freq=5, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)#, update_freq='epoch'
    #history = model.fit(X_tr, Y_train,epochs=nb_epoch,batch_size=batch_size,validation_split=0.1,
                        #callbacks=[model_checkpoint, predcall,tb],verbose=1)#early_stopping, 
    history = model.fit(X_tr, Y_train, epochs=nb_epoch, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint, predcall,tb], validation_data=(X_te, Y_test))#early_stopping,
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
    score = model.evaluate(X_tr, Y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f' %
          (score[0], score[1]))
    prediction = model.predict(X_tr,verbose=0)
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.train_prediction.pkl'.format(hyperparams_name)), 'wb'))
    score = model.evaluate(
        X_te, Y_test, batch_size=batch_size, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f' %
          (score[0], score[1]))
    prediction = model.predict(X_te,verbose=0)
    pickle.dump((prediction), open(os.path.join(
        path_result, '{}.test_prediction.pkl'.format(hyperparams_name)), 'wb'))
    
    f, ax1 = plt.subplots(1, 1)
    f.set_size_inches(12, 6)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_title('Training & Validation Loss',size=25)
    ax1.set_ylabel('Loss',size=20)
    #ax1.set_ylim(0,20)
    ax1.set_xlabel('Epoch',size=20)
    rmse_train=ax1.plot(history.history['loss'][1:])
    rmse_val=ax1.plot(history.history['val_loss'][1:])
    ax1.legend( ( rmse_train[0], rmse_val[0]), ('Training','Validation'),fontsize=15)    
    f.savefig(os.path.join(path_pic, '{}.loss.png'.format(hyperparams_name)))
    f, ax2 = plt.subplots(1, 1)
    f.set_size_inches(12, 6)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_title('Prediction',size=25)
    ax2.set_ylabel('In(Streamflow of Ganges)',size=20)
    ax2.set_xlabel('Days',size=20)
    #ax2.set_ylim(0,800)
    pre=ax2.plot(prediction)
    truth=ax2.plot(Y_test.values)

    ax2.legend(( pre[0], truth[0]), ('Prediction','True Value'),fontsize=15)
    f.savefig(os.path.join(path_pic, '{}.prediction.png'.format(hyperparams_name)))
