

"""
Created on Sat December 15 21:04:06 2018

@author: Yong Zhuang
"""
from __future__ import division

import six
from keras.models import Model
from keras.layers import Permute  
from keras.layers import TimeDistributed 
from keras.layers import Flatten 
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import get_source_inputs
import fusion_net as fusion


def TVGG16(input_tensor=None,
          input_shape=None,
          pooling=None,drop = None,prefix=''):
    """Instantiates the TimeDistributed VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor              
    print ('img_input shape: ',img_input.shape)   
    # Block 1
    x = TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block1_conv1')(img_input) 
    x = TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same',data_format = 'channels_first'),name=prefix+'block1_conv2')(x) 
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block1_pool')(x)
             
    print ('x shape: ',x.shape)   
    # Block 2
    x = TimeDistributed(Conv2D(32, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block2_conv1')(x)
    x = TimeDistributed(Conv2D(32, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block2_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block2_pool')(x)
             
    print ('x shape: ',x.shape)   
    # Block 3
    x = TimeDistributed(Conv2D(18, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block3_conv1')(x)
    x = TimeDistributed(Conv2D(18, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block3_conv2')(x)
    x = TimeDistributed(Conv2D(18, 3,
                      activation='relu',
                      padding='same',data_format = 'channels_first'),name=prefix+'block3_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block3_pool')(x)
             
    print ('x shape: ',x.shape)  
    ''' 
    # Block 4
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block4_conv1')(x)
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block4_conv2')(x)
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block4_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block4_pool')(x)
             
    print ('x shape: ',x.shape)   
    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv1')(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv2')(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block5_pool')(x)
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    x = TimeDistributed(Flatten())(x)
    if drop is not None:
        x = Dropout(drop)(x)
    # Create model.
    model = Model(inputs, x, name='tvgg16')
    #model.summary()
    return model            

def TVGG16_v2(input_tensor=None,
          input_shape=None,
          pooling=None,drop = None,prefix=''):
    """Instantiates the TimeDistributed VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor              
    # Block 1
    #x = TimeDistributed(Conv2D(18, 3,
                      #activation='relu',
                      #padding='same', data_format = 'channels_first'),name=prefix+'block1_conv1')(img_input) 
    #x = TimeDistributed(Conv2D(18, 3, activation='relu', padding='same',data_format = 'channels_first'),name=prefix+'block1_conv2')(x) 
    #x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block1_pool')(x)
               
    # Block 2
    x = TimeDistributed(Conv2D(32, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block2_conv1')(img_input)
    x = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same', data_format = 'channels_first'),name=prefix+'block2_conv2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block2_pool')(x)
               
    # Block 3
    x = TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block3_conv1')(x)
    x = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', data_format = 'channels_first'),name=prefix+'block3_conv2')(x)
    #x = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same',data_format = 'channels_first'),name=prefix+'block3_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block3_pool')(x)
             
    # Block 4 
    x = TimeDistributed(Conv2D(128, 3,
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block4_conv1')(x)
    x = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', data_format = 'channels_first'),name=prefix+'block4_conv2')(x)
    #x = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', data_format = 'channels_first'),name=prefix+'block4_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block4_pool')(x)
          
    # Block 5    
    '''  
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv1')(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv2')(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', data_format = 'channels_first'),name=prefix+'block5_conv3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block5_pool')(x)
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    x = TimeDistributed(Flatten())(x)
    if drop is not None:
        x = Dropout(drop)(x)
    # Create model.
    model = Model(inputs, x, name='tvgg16')
    #model.summary()
    return model              
def net(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look, o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    print (o_shape,f_shape)
    model = None
    pl = 'max'
    dp = None
    input0 = Input(shape=o_shape)
    input1 = Input(shape=f_shape) 
    input2 = Input(shape=f_shape)
    input_fusion = concatenate([input1, input2],axis = 1)
    
    inp0 = Permute((2,1,3,4))(input0)
    inp_fusion = Permute((2,1,3,4))(input_fusion)
    
    print ('inp0 shape',inp0.shape)
    print ('inp_fusion shape',inp_fusion.shape)
    vgg0 = TVGG16(input_tensor = inp0, pooling = pl,drop=dp,prefix='vgg0')
    
    vgg1 = TVGG16(input_tensor = inp_fusion, pooling = pl,drop=dp,prefix='vgg1')
    
    
    for layer in vgg0.layers:
        layer.trainable = True
    x0 = vgg0.output
    print ('x0 shape',x0.shape)
    x0 = LSTM(256, return_sequences=True, dropout=dp, name ='lstm1')(x0)
    x0 = Flatten()(x0)
    print ('x0 shape',x0.shape)
    for layer in vgg1.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = vgg1.output
    print ('x1 shape',x1.shape)
    x1 = LSTM(256, return_sequences=True, dropout=dp, name ='lstm2')(x1)
    x1 = Flatten()(x1)
    print ('x1 shape',x1.shape)
    input3 = Input((external_dim,))    
    f = fusion.net_v2(x0,x1,input3)    
    model = Model(inputs=[input0,input1,input2,input3], outputs=f)      
    model.summary()
    
    return model             
def net_v2(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look, o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    print (o_shape,f_shape)
    model = None
    pl = 'max'
    dp = None
    input0 = Input(shape=o_shape)
    input1 = Input(shape=f_shape) 
    input2 = Input(shape=f_shape)
    input_fusion = concatenate([input1, input2],axis = 1)
    
    inp0 = Permute((2,1,3,4))(input0)
    inp_fusion = Permute((2,1,3,4))(input_fusion)
    
    print ('inp0 shape',inp0.shape)
    print ('inp_fusion shape',inp_fusion.shape)
    vgg0 = TVGG16_v2(input_tensor = inp0, pooling = pl,drop=dp,prefix='vgg_v20')
    
    vgg1 = TVGG16_v2(input_tensor = inp_fusion, pooling = pl,drop=dp,prefix='vgg1')
    
    
    for layer in vgg0.layers:
        layer.trainable = True
    x0 = vgg0.output
    print ('x0 shape',x0.shape)
    x0 = LSTM(32, return_sequences=True, dropout=dp, name ='lstm1_1')(x0)
    x0 = LSTM(16, return_sequences=True, dropout=dp, name ='lstm1_2')(x0)
    #x0 = LSTM(8, return_sequences=True, dropout=dp, name ='lstm1')(x0)
    x0 = Flatten()(x0)
    for layer in vgg1.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = vgg1.output
    print ('x1 shape',x1.shape)
    x1 = LSTM(32, return_sequences=True, dropout=dp, name ='lstm2_1')(x1)
    x1 = LSTM(16, return_sequences=True, dropout=dp, name ='lstm2_2')(x1)
    #x1 = LSTM(8, return_sequences=True, dropout=dp, name ='lstm2')(x1)
    x1 = Flatten()(x1)
    input3 = Input((external_dim,))    
    f = fusion.net_v2(x0,x1,input3)    
    model = Model(inputs=[input0,input1,input2,input3], outputs=f)      
    model.summary()
    
    return model
