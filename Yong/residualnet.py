"""
Created on Sat Sep 7 21:04:06 2018

@author: Yong Zhuang
"""

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape,
    Flatten
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


def _bn_relu_conv(nb_filter=3, strides=(1, 1), kernel_size=(3,3), is_3D=False, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        if is_3D:
            return Conv3D(padding="same", strides=strides, filters=nb_filter, kernel_size=kernel_size)(activation)
        else:
            return Conv2D(padding="same", strides=strides, filters=nb_filter, kernel_size=kernel_size)(activation)
    return f

def _residual_unit(nb_filter=3, strides=(1, 1), kernel_size=(3,3), is_3D=False, bn=False):
    def f(input):
        residual = _bn_relu_conv(nb_filter, strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=bn)(input)
        residual = _bn_relu_conv(nb_filter, strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=bn)(residual)
        return _shortcut(input, residual)
    return f

def ResUnits(residual_unit=3, nb_filter=3, repetations=1, is_3D=False, batchNormalization=False):
    def f(input):
        for i in range(repetations):
            strides = (1, 1)
            kernel_size=(3,3)
            if is_3D:
                strides = (1, 1, 1)
                kernel_size = (2,3,3) # depth, height and width
            input = residual_unit(nb_filter=nb_filter,
                                  strides=strides, kernel_size=kernel_size, is_3D=is_3D, bn=batchNormalization)(input)
        return input
    return f

def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3, is_3D=False, batchNormalization=False):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf

            if is_3D:
                # Conv1
                input = Input(shape=(nb_flow, len_seq, map_height, map_width))
                conv1 = Conv3D (padding="same", filters=64, kernel_size=(2, 3, 3))(input)  
                # [nb_residual_unit] Residual Units
                residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit, is_3D=is_3D, batchNormalization=batchNormalization)(conv1)      
                # Conv2
                activation = Activation('relu')(residual_output)
                conv2 = Conv3D(padding="same", filters=nb_flow, kernel_size=(2, 3, 3))(activation)
            else:
                # Conv1
                input = Input(shape=(nb_flow * len_seq, map_height, map_width))    
                conv1 = Conv2D (padding="same", filters=64, kernel_size=(3, 3))(input)   
                # [nb_residual_unit] Residual Units
                residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit, is_3D=is_3D, batchNormalization=batchNormalization)(conv1)                 
                # Conv2
                activation = Activation('relu')(residual_output)
                conv2 = Conv2D(padding="same", filters=nb_flow, kernel_size=(3, 3))(activation)               
            main_inputs.append(input)
            outputs.append(conv2)
    main_output = outputs[0]
    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    #main_output = Activation('tanh')(main_output)
    if is_3D:
        conv3 = Conv3D (padding="valid", filters=32, kernel_size=(3, 3, 3))(input)   
    else:
        conv3 = Conv2D (padding="valid", filters=32, kernel_size=(3, 3))(main_output)
    sub_conv3 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid')(conv3)
    flat = Flatten()(sub_conv3)    
    flow = Dense(units=100)(flat)
    flow = Dense(units=50)(flat)
    flow = Dense(units=1)(flat)
    flow = Activation('relu')(flow)
    model = Model(inputs=main_inputs, outputs=flow)

    return model