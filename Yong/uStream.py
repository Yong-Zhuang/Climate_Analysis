import numpy as np 
import pandas as pd
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as keras
from netCDF4 import Dataset
np.random.seed(1337) 
import keras
from keras import backend as K
from sklearn import metrics

dirname='../Data/'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
#Root Mean Squared Error
def get_metrics(y, pred):
    m_mae = metrics.mean_absolute_error(y, pred)
    m_rmse = metrics.mean_squared_error(y, pred)** 0.5
    m_r2 = metrics.r2_score(y, pred) 
    return m_mae,m_rmse,m_r2
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5
#Root Mean Squared Logarithmic Error
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(tf.log1p(y_pred) - tf.log1p(y_true))))
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def unet(input_size,nlags,is_3D=False):
    if is_3D:
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
    else:
        input1 = Input(input_size)
        input2 = Input(input_size)
        input3 = Input((nlags,))
        my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input1)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv1)
        pool1 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv1)

        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv2)
        pool2 = AveragePooling2D(pool_size=(2, 2), data_format = 'channels_first')(conv2)

        conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool2)
        conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv3)
        pool3 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv3)

        conv12 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input2)
        conv12 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv12)
        pool12 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv12)

        conv22 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool12)
        conv22 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv22)
        pool22 = AveragePooling2D(pool_size=(2, 2), data_format = 'channels_first')(conv22)

        conv32 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool22)
        conv32 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv32)
        pool32 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv32)

    #    m1 = add([conv1,conv12])
    #    m1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m1)
    #    m1 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m1)
    #    m1 = AveragePooling2D(pool_size=(3,2,2), data_format = 'channels_first')(m1)
    #    
    #    m2 = add([pool1,pool12])  
    #    m2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m2)
    #    m2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m2)
    #    m2 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(m2)
    #
    #    m3 = add([m2,pool2,pool22])  
    #    m3 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m3)
    #    m3 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format = 'channels_first')(m3)
    #    m3 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(m3)    

        m4 = add([pool3,pool32])  
        m4 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
        m4 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
        m4 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(m4)
    #    m4 = Dropout(0.5)(m4)            
        f = Flatten()(m4)
        f = Dense(32,activation='relu')(f)
        f = Dropout(0.5)(f)    
        f = Dense(1,activation ='relu')(f)    

        q = Dense(1)(input3)    
        f = add([q,f])

        model = Model(inputs=[input1,input2,input3], outputs=f)
        #model.summary()

        return model

input_size = (1,15,64,128)
#input_size = (15,64,128)
Nlags = 15

fname_param = dirname+'3d_mean_spread_best_parameters_{}.hdf5'.format('1_9')
early_stopping = EarlyStopping(monitor='val_loss', patience=10,mode='auto')
model_checkpoint = ModelCheckpoint(fname_param, verbose=1, save_best_only=True, mode='min',save_weights_only=True)
  

#inx = Dataset('../Data/total_forecast_precipitation_mean_spread_input.nc','r')
inx = np.load(dirname+'total_forecast_precipitation_mean_spread_input.npy')
Px_mean = inx[:,:,:,:,0]
Px_mean = Px_mean[:,np.newaxis,:,:,:]
Px_mean = np.moveaxis(Px_mean,4,2)
#Px_mean = np.moveaxis(Px_mean,3,1)
Px_spread = inx[:,:,:,:,1]
Px_spread = Px_spread[:,np.newaxis,:,:,:]
Px_spread = np.moveaxis(Px_spread,4,2)
#Px_spread = np.moveaxis(Px_spread,3,1)
print('mean shape: ',Px_mean.shape)


Qx = pd.read_csv(dirname+'X_Ganges.csv')
Qx = np.array(Qx.loc[:,['Q_-1','Q_0']])

Qy = pd.read_csv(dirname+'Y_Ganges.csv')
Qy = np.array(Qy.loc[:,'Q_15'])

test_train = 3060
train_validate = 3825
Xtrain = [Px_mean[0:test_train,:,:,:,:],Px_spread[0:test_train,:,:,:,:],Qx[0:test_train]] 
Xval = [Px_mean[test_train:train_validate,:,:,:,:],Px_spread[test_train:train_validate,:,:,:,:],Qx[test_train:train_validate]] 
Xtest = [Px_mean[train_validate:,:,:,:,:],Px_spread[train_validate:,:,:,:,:],Qx[train_validate:]] 

#Xtrain = [Px_mean[0:test_train,:,:,:],Px_spread[0:test_train,:,:,:],Qx[0:test_train]] 
#Xval = [Px_mean[test_train:train_validate,:,:,:],Px_spread[test_train:train_validate,:,:,:],Qx[test_train:train_validate]] 
#Xtest = [Px_mean[train_validate:,:,:,:],Px_spread[train_validate:,:,:,:],Qx[train_validate:]] 
trainingQ = Qy[0:test_train]
valQ =  Qy[test_train:train_validate]
testQ = Qy[train_validate:]

time,nlags=Qx.shape
model = unet(input_size,nlags,True)
model.compile(loss='mae', optimizer='adam', metrics=[rmse,r_squared])

history = model.fit(Xtrain, trainingQ,
                epochs=1000,
                batch_size=30,
                validation_data=(Xval,valQ),
                callbacks=[early_stopping,model_checkpoint],
                verbose=1)

model.load_weights(fname_param)
prediction = model.predict(Xtrain,verbose=0)
m_mae,m_rmse,m_r2 = get_metrics(trainingQ,prediction)
print('Train score mae: %.6f rmse: %.6f r-square: %.6f' %
      (m_mae,m_rmse,m_r2))

prediction = model.predict(Xtest,verbose=0)
m_mae,m_rmse,m_r2 = get_metrics(testQ,prediction)
print('Test score mae: %.6f rmse: %.6f r-square: %.6f' %
      (m_mae,m_rmse,m_r2))

Qhat = model.predict(Xtest, batch_size=1, verbose=0)
Q=pd.concat([pd.DataFrame(Qhat),pd.DataFrame(testQ)],axis=1)
Q.columns = ['Predicted','Observed']



    
    