from __future__ import division

import os
import six
from keras import Sequential
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
          pooling=None,drop = None):
    # Block 1
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv1', data_format = 'channels_first'),input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv2', data_format = 'channels_first')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format = 'channels_first')))
             
 
    # Block 2
    model.add(TimeDistributed(Conv2D(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv1', data_format = 'channels_first')))
    model.add(TimeDistributed(Conv2D(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv2', data_format = 'channels_first')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format = 'channels_first')))
             

    # Block 3
    model.add(TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv1', data_format = 'channels_first')))
    model.add(TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv2', data_format = 'channels_first')))
    model.add(TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv3', data_format = 'channels_first')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format = 'channels_first')))
             
 
    # Block 4
    model.add(TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv1', data_format = 'channels_first')))
    model.add(TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv2', data_format = 'channels_first')))
    model.add(TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv3', data_format = 'channels_first')))
    #model.add(TimeDistributed(MaxPooling2D(3, strides=(2, 2), name='block4_pool', data_format = 'channels_first')))
             
  
    # Block 5
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv1'))(x)
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv2'))(x)
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv3'))(x)
    #x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)

    if drop is not None:
        x = Dropout(drop)(x)
    model.summary()
    return model 
def TVGG16_v1(input_tensor=None,
          input_shape=None,
          pooling=None,drop = None):
    # Block 1
    img_input = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv1', data_format = 'channels_first'))(img_input) 
    x = TimeDistributed(Conv2D(64, 3,
                      activation='relu',
                      padding='same',
                      name='block1_conv2', data_format = 'channels_first'))(x) 
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format = 'channels_first'))(x)
 
    # Block 2
    x = TimeDistributed(Conv2D(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv1', data_format = 'channels_first'))(x)
    x = TimeDistributed(Conv2D(128, 3,
                      activation='relu',
                      padding='same',
                      name='block2_conv2', data_format = 'channels_first'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format = 'channels_first'))(x)
             
             

    # Block 3
    x = TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv1', data_format = 'channels_first'))(x)
    x = TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv2', data_format = 'channels_first'))(x)
    x = TimeDistributed(Conv2D(256, 3,
                      activation='relu',
                      padding='same',
                      name='block3_conv3', data_format = 'channels_first'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format = 'channels_first'))(x)
             
 
    # Block 4
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv1', data_format = 'channels_first'))(x)
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv2', data_format = 'channels_first'))(x)
    x = TimeDistributed(Conv2D(512, 3,
                      activation='relu',
                      padding='same',
                      name='block4_conv3', data_format = 'channels_first'))(x)
    #x = TimeDistributed(MaxPooling2D(3, strides=(2, 2), name='block4_pool', data_format = 'channels_first'))(x)
    #model.add(TimeDistributed(MaxPooling2D(3, strides=(2, 2), name='block4_pool', data_format = 'channels_first')))
             
  
    # Block 5
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv1'))(x)
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv2'))(x)
    #x = TimeDistributed(Conv2D(512, (3, 3),
                      #activation='relu',
                      #padding='same',
                      #name='block5_conv3'))(x)
    #x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)

    if drop is not None:
        x = Dropout(drop)(x)
    model = Model(img_input, x, name='tvgg16')
    model.summary()
    return model 
if __name__== "__main__": 
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    model = TVGG16_v1(input_shape=(2,10,64,12))