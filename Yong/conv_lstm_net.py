"""
Created on Sat December 15 21:04:06 2018

@author: Yong Zhuang
"""

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape,
    Flatten,
    concatenate,
    TimeDistributed
)
from keras.models import Model
import numpy as np
import pylab as plt

def convLSTM_net(conf=(15, 1, 32, 32), external_dim=8,  kernel_size=(3, 3), filters=40, nb_stack=1, batchNormalization=False, regularization=True): 
    main_inputs = [] 
    timesteps, channels, map_height, map_width = conf
    input = Input(shape=(timesteps, channels, map_height, map_width)) 
    if regularization:
        convlstm_output = ConvLSTM2D(filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),padding='same', return_sequences=True,data_format='channels_first')(input)#,recurrent_regularizer=keras.regularizers.l1(0.01) 
    else:
        convlstm_output = ConvLSTM2D(filters=filters, kernel_size=kernel_size,padding='same', return_sequences=True,data_format='channels_first')(input)
        
    if batchNormalization:
        convlstm_output = BatchNormalization(mode=0, axis=1)(convlstm_output)
    for i in range(nb_stack):
        if regularization:
            convlstm_output = ConvLSTM2D(filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),padding='same', return_sequences=True,data_format='channels_first')(convlstm_output)  
        else:
            convlstm_output = ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same', return_sequences=True,data_format='channels_first')(convlstm_output)
        if batchNormalization:
            convlstm_output = BatchNormalization(mode=0, axis=1)(convlstm_output)
    convlstm_output = TimeDistributed(Flatten())(convlstm_output)  
    convlstm_output = TimeDistributed(Dense(units=10,activation='relu'))(convlstm_output) 
    convlstm_output = TimeDistributed(Dense(units=1,activation='relu'))(convlstm_output) 
    convlstm_output = Flatten()(convlstm_output) 
    main_inputs.append(input)
    init_input = Input(shape=(external_dim,))
    main_inputs.append(init_input)
    main_output = concatenate([init_input, convlstm_output])
    main_output = Dense(units=10,activation='relu')(main_output)
    out = Dense(units=1,activation='relu')(main_output)
    model = Model(inputs=main_inputs, outputs=out)
    return model
