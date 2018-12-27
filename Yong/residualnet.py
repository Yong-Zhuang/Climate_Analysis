"""
Created on Sat Sep 7 21:04:06 2018

@author: Yong Zhuang
"""

from __future__ import print_function
import keras
from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape,
    Flatten,
    concatenate
)
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    print (input.shape)
    print (residual.shape)
    return add([input, residual])


def _bn_relu_conv(filters=3, strides=(1, 1), kernel_size=(3,3), is_3D=False, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        if is_3D:
            return Conv3D(padding="same", strides=strides, filters=filters, kernel_size=kernel_size,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(activation)#, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        else:
            return Conv2D(padding="same", strides=strides, filters=filters, kernel_size=kernel_size,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(activation)#, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    return f 

def _residual_unit(filters=64, strides=(1, 1), kernel_size=(3,3), is_3D=False, bn=False):
    def f(input):
        residual = _bn_relu_conv(filters, strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=bn)(input)
        #residual = _bn_relu_conv(nb_filter, strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=bn)(residual)
        return _shortcut(input, residual)
    return f

def ResUnits(residual_unit=3, filters=64, kernel_size=(3,3), repetations=1, is_3D=False, batchNormalization=False):
    def f(input):
        for i in range(repetations):
            strides = (1, 1)
            if is_3D:
                strides = (1, 1, 1)
            input = residual_unit(filters=filters, strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=batchNormalization)(input)
        return input
    return f

def stresnet(conf=(15, 1, 32, 32), external_dim=5, kernel_size=(3,3), filters=64, nb_resunit=0,nb_conv=128, is_3D=False, batchNormalization=False):
    '''
    conf = (timesteps, channels, map_height, map_width) 
    external_dim: non-spatio-temporal features   
    kernel_size: kernel size
    filters: # of kernels
    nb_resunit: # of residual layers
    nb_conv: # of convolutional layers
    is_3D: using 3D convolution
    batchNormalization: using batch Normalization
    '''
    # main input
    main_inputs = []
    if conf is not None:
        timesteps, channels, map_height, map_width = conf
        if is_3D:
            #input = Input(shape=(nb_flow, len_seq, map_height, map_width))
            input = Input(shape=(timesteps, channels, map_height, map_width))
            conv_output = Conv3D (padding="same", filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(input)  
            if nb_resunit>0:
                conv_output = ResUnits(_residual_unit, filters=filters, kernel_size=kernel_size,
                          repetations=nb_resunit, is_3D=is_3D, batchNormalization=batchNormalization)(conv_output)
            elif nb_conv>0:
                for i in range(nb_conv):
                    conv_output = Conv3D (padding="same", filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(conv_output)
            activation = Activation('relu')(conv_output)
            conv2 = Conv3D(padding="same", filters=channels, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(activation)
        else:
            input = Input(shape=(timesteps * channels, map_height, map_width))    
            conv_output = Conv2D (padding="same", filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(input)  
            if nb_resunit>0:
                conv_output = ResUnits(_residual_unit, filters=filters, kernel_size=kernel_size,
                          repetations=nb_resunit, is_3D=is_3D, batchNormalization=batchNormalization)(conv_output)
            elif nb_conv>0:
                for i in range(nb_conv):
                    conv_output = Conv2D (padding="same", filters=filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(conv_output) 
            activation = Activation('relu')(conv_output)
            conv2 = Conv2D(padding="same", filters=channels, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(activation)             
        main_inputs.append(input)
    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        h1 = Dense(units=10)(external_input)
        h1 = Activation('relu')(h1)
        #h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        h2 = Dense(units=2)(h1)
        #activation = Activation('relu')(h1)
        external_output = Activation('relu')(h2)
        #external_output = Reshape((nb_flow, map_height, map_width))(activation)
    else:
        external_output = []   

    flat = Flatten()(conv2)    
    flat = Activation('relu')(flat)
    if external_dim != None and external_dim > 0:
        conc = concatenate([external_output, flat])
    else:
        conc = flat
    output = Dense(units=10,activation='relu')(conc)
    out = Dense(units=1,activation='relu')(output)
    model = Model(inputs=main_inputs, outputs=out)
    return model
