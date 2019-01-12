import numpy as np 
import pandas as pd
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as keras
from netCDF4 import Dataset
np.random.seed(1) # NumPy
import random
random.seed(2) # Python
from tensorflow import set_random_seed
set_random_seed(3) # Tensorflow
import keras

def unet(input_size,nlags):
    input1 = Input(input_size)
    input2 = Input(input_size)
    input3 = Input((nlags,))
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input1)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv1)
    pool1 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv1)

    conv2 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool1)
    conv2 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv2)
    pool2 = AveragePooling3D(pool_size=(1, 2, 2), data_format = 'channels_first')(conv2)
    
    conv3 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool2)
    conv3 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv3)
    pool3 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv3)
        
    conv12 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input2)
    conv12 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv12)
    pool12 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv12)

    conv22 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool12)
    conv22 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv22)
    pool22 = AveragePooling3D(pool_size=(1, 2, 2), data_format = 'channels_first')(conv22)
    
    conv32 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool22)
    conv32 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv32)
    pool32 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv32)

#    m1 = add([conv1,conv12])
#    m1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m1)
#    m1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m1)
#    m1 = AveragePooling3D(pool_size=(3,2,2), data_format = 'channels_first')(m1)
#    
#    m2 = add([pool1,pool12])  
#    m2 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m2)
#    m2 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m2)
#    m2 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(m2)
#
#    m3 = add([m2,pool2,pool22])  
#    m3 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m3)
#    m3 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m3)
#    m3 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(m3)    
        
    m4 = add([pool3,pool32])  
    m4 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
    m4 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
    m4 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(m4)
#    m4 = Dropout(0.5)(m4)            
    f = Flatten()(m4)
    f = Dense(32,activation='relu')(f)
    f = Dropout(0.5)(f)    
    f = Dense(1,activation ='relu')(f)    

    q = Dense(1)(input3)    
    f = add([q,f])
 
    model = Model(inputs=[input1,input2,input3], outputs=f)
    model.summary()
    
    return model

input_size = (1,15,64,128)
Nlags = 15

fname_param = '/media/smalldave/Storage/GBM/LSTM/3d_mean_spread_best_parameters_{}.hdf5'.format('1_9')
early_stopping = EarlyStopping(monitor='val_loss', patience=10,mode='auto')
model_checkpoint = ModelCheckpoint(fname_param, verbose=1, save_best_only=True, mode='min',save_weights_only=True)
  

inx = Dataset('/media/smalldave/Storage/GBM/LSTM/total_forecast_precipitation_mean_spread_input.nc')

Px_mean = inx.variables['precipitation'][:,:,:,:,0]
Px_mean = Px_mean[:,np.newaxis,:,:,:]
Px_mean = np.moveaxis(Px_mean,4,2)
Px_spread = inx.variables['precipitation'][:,:,:,:,1]
Px_spread = Px_spread[:,np.newaxis,:,:,:]
Px_spread = np.moveaxis(Px_spread,4,2)
Qx = pd.read_csv('/media/smalldave/Storage/GBM/LSTM/X_Ganges.csv')
Qx = np.array(Qx.loc[:,['Q_-1','Q_0']])

Qy = pd.read_csv('/media/smalldave/Storage/GBM/LSTM/Y_Ganges.csv')
Qy = np.array(Qy.loc[:,'Q_15'])

test_train = 3060
train_validate = 3825
Xtrain = [Px_mean[0:test_train,:,:,:,:],Px_spread[0:test_train,:,:,:,:],Qx[0:test_train]] 
Xval = [Px_mean[test_train:train_validate,:,:,:,:],Px_spread[test_train:train_validate,:,:,:,:],Qx[test_train:train_validate]] 
Xtest = [Px_mean[train_validate:,:,:,:,:],Px_spread[train_validate:,:,:,:,:],Qx[train_validate:]] 

trainingQ = Qy[0:test_train]
valQ =  Qy[test_train:train_validate]
testQ = Qy[train_validate:]

time,nlags=Qx.shape
model = unet(input_size,nlags)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history = model.fit(Xtrain, trainingQ,
                epochs=1000,
                batch_size=30,
                validation_data=(Xval,valQ),
                callbacks=[early_stopping,model_checkpoint],
                verbose=1)

model.load_weights(fname_param)
score_train = model.evaluate(Xtrain, trainingQ, batch_size=1, verbose=0)
print('Train score: %.6f rmse (norm): %.6f' %
      (score_train[0], score_train[1]))

score_test = model.evaluate(Xtest, testQ, batch_size=1, verbose=0)
print('Test score: %.6f rmse (norm): %.6f' %
      (score_test[0], score_test[1]))
print score_test[0]/score_train[0]

Qhat = model.predict(Xtest, batch_size=1, verbose=0)
Q=pd.concat([pd.DataFrame(Qhat),pd.DataFrame(testQ)],axis=1)
Q.columns = ['Predicted','Observed']



    
    