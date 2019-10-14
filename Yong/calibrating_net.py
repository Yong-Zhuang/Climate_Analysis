import numpy as np 
import pandas as pd
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as keras
from netCDF4 import Dataset
import keras

def net(conf,nlags,kernel_size,is_3D=False):
    timesteps, channels, map_height, map_width = conf
    if is_3D:
        input1 = Input(shape=(channels, timesteps, map_height, map_width))
        input2 = Input(shape=(channels, timesteps, map_height, map_width)) 
        print ((channels, timesteps, map_height, map_width),nlags)
    #input1 = Input(input_size)
    #input2 = Input(input_size)
        input3 = Input((nlags,))
        my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
        conv1 = Conv3D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input1)
        conv1 = Conv3D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv1)
        pool1 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv1)

        conv2 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool1)
        conv2 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv2)
        pool2 = AveragePooling3D(pool_size=(1, 2, 2), data_format = 'channels_first')(conv2)

        conv3 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool2)
        conv3 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv3)
        pool3 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv3)

        conv12 = Conv3D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input2)
        conv12 = Conv3D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv12)
        pool12 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(conv12)

        conv22 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool12)
        conv22 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv22)
        pool22 = AveragePooling3D(pool_size=(1, 2, 2), data_format = 'channels_first')(conv22)

        conv32 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool22)
        conv32 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv32)
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
        m4 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
        m4 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
        m4 = AveragePooling3D(pool_size=(1,2,2), data_format = 'channels_first')(m4)
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
    else:
        input1 = Input(shape=(timesteps * channels, map_height, map_width)) 
        input2 = Input(shape=(timesteps * channels, map_height, map_width))
        input3 = Input((nlags,))
        my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
        conv1 = Conv2D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input1)
        conv1 = Conv2D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv1)
        pool1 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv1)

        conv2 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool1)
        conv2 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv2)
        pool2 = AveragePooling2D(pool_size=(2, 2), data_format = 'channels_first')(conv2)

        conv3 = Conv2D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool2)
        conv3 = Conv2D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv3)
        pool3 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv3)

        conv12 = Conv2D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(input2)
        conv12 = Conv2D(64, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv12)
        pool12 = AveragePooling2D(pool_size=(2,2), data_format = 'channels_first')(conv12)

        conv22 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool12)
        conv22 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv22)
        pool22 = AveragePooling2D(pool_size=(2, 2), data_format = 'channels_first')(conv22)

        conv32 = Conv2D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(pool22)
        conv32 = Conv2D(16, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(conv32)
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
        m4 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
        m4 = Conv2D(32, kernel_size, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init)(m4)
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