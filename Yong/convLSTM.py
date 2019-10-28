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
from keras.layers import ConvLSTM2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import get_source_inputs
import fusion_net as fusion
def convLSTM(input_tensor=None,input_shape=None, final_pooling=None,drop = None,prefix=''):
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor               
    # Block 1
    x = ConvLSTM2D(filters=64, kernel_size=3,padding='same', activation="relu", return_sequences=True,data_format='channels_first',name=prefix+'block1_convlstm')(img_input)
    x = BatchNormalization(axis=2, name=prefix+'block1_bn')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block1_pool')(x)
                 
    # Block 2
    '''
    x = ConvLSTM2D(filters=32, kernel_size=3,padding='same', activation="relu", return_sequences=True,data_format='channels_first',name=prefix+'block2_convlstm')(x)
    x = BatchNormalization(axis=2, name=prefix+'block2_bn')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block2_pool')(x)             
    # Block 13
    x = ConvLSTM2D(filters=64, kernel_size=3,padding='same', activation="relu", return_sequences=True,data_format='channels_first',name=prefix+'block3_convlstm')(x)
    x = BatchNormalization(axis=2, name=prefix+'block3_bn')(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), data_format = 'channels_first'),name=prefix+'block3_pool')(x)
    
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    if final_pooling == 'avg':
        x = GlobalAveragePooling3D()(x)
    elif final_pooling == 'max':
        x = GlobalMaxPooling3D()(x)

    if drop is not None:
        x = Dropout(drop)(x)
    # Create model.
    model = Model(inputs, x, name='convLSTM')
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
    
    cl1 = convLSTM(input_tensor = inp0, final_pooling = pl,drop=dp,prefix='cl1')
    
    cl2 = convLSTM(input_tensor = inp_fusion, final_pooling = pl,drop=dp,prefix='cl2')
    
    
    for layer in cl1.layers:
        layer.trainable = True
    x0 = cl1.output
    print ('x0 shape',x0.shape)
    for layer in cl2.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = cl2.output
    print ('x1 shape',x1.shape)
    input3 = Input((external_dim,))    
    f = fusion.net_v2(x0,x1,input3)    
    model = Model(inputs=[input0,input1,input2,input3], outputs=f)      
    model.summary()
    return model