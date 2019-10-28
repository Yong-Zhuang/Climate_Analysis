"""ResNet v1, v2, and segmentation models for Keras.
# Reference
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
Reference material for extended functionality:
- [ResNeXt](https://arxiv.org/abs/1611.05431) for Tiny ImageNet support.
- [Dilated Residual Networks](https://arxiv.org/pdf/1705.09914) for segmentation support.
- [Deep Residual Learning for Instrument Segmentation in Robotic Surgery](https://arxiv.org/abs/1703.08580)
  for segmentation support.
Implementation Adapted from: github.com/raghakot/keras-resnet
"""
from __future__ import division

import six
from keras.models import Model
from keras.layers import *
from keras.layers import concatenate
from keras.layers import SpatialDropout3D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import GlobalMaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import get_source_inputs 
import fusion_net as fusion


def _bn_relu(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        x = Conv3D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation function.
    This is the ResNet v2 scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_z = int(round(input_shape[Z_AXIS] / residual_shape[Z_AXIS]))
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv3D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1, 1),
                          strides=(stride_z, stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base)(shortcut)

    return add([shortcut, residual])


def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None, transition_dilation_rates=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv):
    """Builds a residual block with repeating bottleneck blocks.
       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each transition
    """
    if transition_dilation_rates is None:
        transition_dilation_rates = [(1, 1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=(is_first_layer and i == 0),
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
        return x

    return f


def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by stage and block.
    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the paper and keras
    and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base


def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
                residual_unit=_bn_relu_conv):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)

    return f


def bottleneck(filters, stage, block, transition_strides=(1, 1),
               dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
               residual_unit=_bn_relu_conv):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input_feature):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_feature)
        else:
            x = residual_unit(filters=filters, kernel_size=(1, 1, 1),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_feature)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters * 4, kernel_size=(1, 1, 1),
                          conv_name_base=conv_name_base + '2c',
                          bn_name_base=bn_name_base + '2c')(x)

        return _shortcut(input_feature, x)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global Z_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        Z_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        Z_AXIS = 2
        ROW_AXIS = 3
        COL_AXIS = 4


def _string_to_function(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def ResNet(input_shape=None, block='bottleneck', residual_unit='v2', repetitions=None,
           initial_filters=64, activation='softmax', input_tensor=None, dropout=None,
           transition_dilation_rate=(1, 1, 1), initial_strides=(2, 2, 2), initial_kernel_size=(7, 7, 7),
           initial_pooling='max', final_pooling='max'):
    """Builds a custom ResNet like architecture. Defaults to ResNet50 v2.
    Args:
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False
        block: The block function to use. This is either `'basic'` or `'bottleneck'`.
            The original paper used `basic` for layers < 50.
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved.
            Default of None implies the ResNet50v2 values of [3, 4, 6, 3].
        residual_unit: the basic residual unit, 'v1' for conv bn relu, 'v2' for bn relu conv.
            See [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
            for details.
        dropout: None for no dropout, otherwise rate of dropout from 0 to 1.
            Based on [Wide Residual Networks.(https://arxiv.org/pdf/1605.07146) paper.
        transition_dilation_rate: Dilation rate for transition layers. For semantic
            segmentation of images use a dilation rate of (2, 2).
        initial_strides: Stride of the very first residual unit and MaxPooling3D call,
            with default (2, 2), set to (1, 1) for small images like cifar.
        initial_kernel_size: kernel size of the very first convolution, (7, 7) for imagenet
            and (3, 3) for small image datasets like tiny imagenet and cifar.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        initial_pooling: Determine if there will be an initial pooling layer,
            'max' for imagenet and None for small image datasets.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        final_pooling: Optional pooling mode for feature extraction at the final model layer
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                3D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns:
        The keras `Model`.
    """
    if repetitions is None:
        repetitions = [3, 4, 6, 3]
    _handle_dim_ordering()
    if input_shape is not None and len(input_shape) != 4:
        raise Exception("Input shape should be a tuple (nb_channels, time_steps, nb_rows, nb_cols)")

    if block == 'basic':
        block_fn = basic_block
    elif block == 'bottleneck':
        block_fn = bottleneck
    elif isinstance(block, six.string_types):
        block_fn = _string_to_function(block)
    else:
        block_fn = block

    if residual_unit == 'v2':
        residual_unit = _bn_relu_conv
    elif residual_unit == 'v1':
        residual_unit = _conv_bn_relu
    elif isinstance(residual_unit, six.string_types):
        residual_unit = _string_to_function(residual_unit)
    else:
        residual_unit = residual_unit

    #img_input = Input(shape=input_shape, tensor=input_tensor)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size, strides=initial_strides)(img_input)
    if initial_pooling == 'max':
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=initial_strides, padding="same")(x)

    block = x
    filters = initial_filters
    #x = SpatialDropout3D(0.5)(x)
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1, 1)] * r
        if transition_dilation_rate == (1, 1, 1):
            transition_strides[0] = (2, 2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)

    # Classifier block
    if final_pooling == 'avg':
        x = GlobalAveragePooling3D()(x)
    elif final_pooling == 'max':
        x = GlobalMaxPooling3D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs=inputs, outputs=x)
    return model

def ResNet4(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 18 layers and v2 residual units
    """
    return ResNet(input_shape, basic_block, repetitions=[1],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)

def ResNet18(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 18 layers and v2 residual units
    """
    return ResNet(input_shape, basic_block, repetitions=[2, 2, 2, 2],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)


def ResNet34(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 34 layers and v2 residual units
    """
    return ResNet(input_shape, basic_block, repetitions=[3, 4, 6, 3],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)


def ResNet50(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 50 layers and v2 residual units
    """
    return ResNet(input_shape, bottleneck, repetitions=[3, 4, 6, 3],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)


def ResNet101(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 101 layers and v2 residual units
    """
    return ResNet(input_shape, bottleneck, repetitions=[3, 4, 23, 3],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)


def ResNet152(input_shape=None,pooling='max',dropout=None,input_tensor=None):
    """ResNet with 152 layers and v2 residual units
    """
    return ResNet(input_shape, bottleneck, repetitions=[3, 8, 36, 3],initial_pooling=pooling,final_pooling=pooling,dropout=dropout, input_tensor=input_tensor)

def net(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look , o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    pl = 'max'
    dp = None
    if mode==4:
        res_net0 = ResNet4(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet4(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet4(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==18:
        res_net0 = ResNet18(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet18(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet18(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==34:
        res_net0 = ResNet34(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet34(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet34(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==50:
        res_net0 = ResNet50(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet50(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet50(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==101:
        res_net0 = ResNet101(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet101(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet101(input_shape=f_shape, pooling=pl,dropout=dp)
    else:
        res_net0 = ResNet152(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet152(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet152(input_shape=f_shape, pooling=pl,dropout=dp)
    for layer in res_net0.layers:
      layer.trainable = True
    x0 = res_net0.output
    for layer in res_net1.layers:
      layer.name = layer.name + str("_1")
      layer.trainable = True
    x1 = res_net1.output
    for layer in res_net2.layers:
      layer.name = layer.name + str("_2")
      layer.trainable = True
    x2 = res_net2.output
    print (x1.shape)
    input3 = Input((external_dim,))
    
    f = fusion.net(x0,x1,x2,input3)    
    model = Model(inputs=[res_net0.input,res_net1.input,res_net2.input,input3], outputs=f)   
    
    
    return model
def net_mt(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look , o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    pl = 'nan'
    dp = None
    input1 = Input(shape=f_shape) 
    input2 = Input(shape=f_shape) 
    input_fusion = concatenate([input1, input2],axis = 2)
    print ('ifusion shape: '+str(input_fusion.shape))
    if mode==4:
        res_net0 = ResNet4(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet4(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==18:
        res_net0 = ResNet18(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet18(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==34:
        res_net0 = ResNet34(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet34(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==50:
        res_net0 = ResNet50(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet50(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==101:
        res_net0 = ResNet101(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet101(input_tensor=input_fusion, pooling=pl,dropout=dp)
    else:
        res_net0 = ResNet152(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet152(input_tensor=input_fusion, pooling=pl,dropout=dp)
    for layer in res_net0.layers:
        layer.trainable = True
    x0 = res_net0.output
    for layer in res_net1.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = res_net1.output
    x3 = Input((external_dim,))
    x4 = Input((external_dim,))
    x5 = Input((external_dim,))
    f = fusion.net_mt2(x0,x1,x3,x4,x5)
    
    model = Model(inputs=[res_net0.input,input1,input2,x3,x4,x5], outputs=f)  
    return model
def net_mt_v1(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look , o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    pl = 'non'
    dp = None
    #input1 = Input(shape=f_shape) 
    #input2 = Input(shape=f_shape) 
    #input_fusion = concatenate([input1, input2],axis = 1)
    if mode==4:
        res_net0 = ResNet4(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet4(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet4(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==18:
        res_net0 = ResNet18(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet18(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet18(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==34:
        res_net0 = ResNet34(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet34(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet34(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==50:
        res_net0 = ResNet50(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet50(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet50(input_shape=f_shape, pooling=pl,dropout=dp)
    elif mode==101:
        res_net0 = ResNet101(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet101(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet101(input_shape=f_shape, pooling=pl,dropout=dp)
    else:
        res_net0 = ResNet152(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet152(input_shape=f_shape, pooling=pl,dropout=dp)
        res_net2 = ResNet152(input_shape=f_shape, pooling=pl,dropout=dp)
    for layer in res_net0.layers:
        layer.trainable = True
    x0 = res_net0.output
    for layer in res_net1.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = res_net1.output
    for layer in res_net2.layers:
        layer.name = layer.name + str("_2")
        layer.trainable = True
    x2 = res_net2.output
    x3 = Input((external_dim,))
    x4 = Input((external_dim,))
    x5 = Input((external_dim,))
    f = fusion.net_mt3(x0,x1,x2,x3,x4,x5)#.net_mt3
    
    model = Model(inputs=[res_net0.input,res_net1.input,res_net2.input,x3,x4,x5], outputs=f)  
    return model

def net_v2(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look , o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    pl = 'max'
    dp = None
    input1 = Input(shape=f_shape) 
    input2 = Input(shape=f_shape) 
    input_fusion = concatenate([input1, input2],axis = 1)
    if mode==4:
        res_net0 = ResNet4(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet4(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==18:
        res_net0 = ResNet18(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet18(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==34:
        res_net0 = ResNet34(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet34(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==50:
        res_net0 = ResNet50(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet50(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==101:
        res_net0 = ResNet101(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet101(input_tensor=input_fusion, pooling=pl,dropout=dp)
    else:
        res_net0 = ResNet152(input_shape=o_shape, pooling=pl,dropout=dp)
        res_net1 = ResNet152(input_tensor=input_fusion, pooling=pl,dropout=dp)
    for layer in res_net0.layers:
        layer.trainable = True
    x0 = res_net0.output
    for layer in res_net1.layers:
        layer.name = layer.name + str("_1")
        layer.trainable = True
    x1 = res_net1.output
    x2 = Input((external_dim,))
    
    f = fusion.net_v3(x0,x1,x2)
    
    model = Model(inputs=[res_net0.input,input1, input2,x2], outputs=f)   
    
    
    return model

def net_mt4(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    len_trend, o_channels, o_height, o_width = o_conf[0]
    len_period, o_channels, o_height, o_width = o_conf[1]
    len_closeness, o_channels, o_height, o_width = o_conf[2]
    len_period, f_channels, f_height, f_width = f_conf[0]
    len_closeness, f_channels, f_height, f_width = f_conf[1]
    o_shape = {}
    f_shape = {}
    o_shape[0] = (o_channels, len_trend , o_height, o_width)
    o_shape[1] = (o_channels, len_period , o_height, o_width)
    o_shape[2] = (o_channels, len_closeness , o_height, o_width)
    f_shape[0] = (f_channels, len_period, f_height, f_width)
    f_shape[1] = (f_channels, len_closeness, f_height, f_width)
    input0_0 = Input(shape=o_shape[0]) 
    input0_1 = Input(shape=o_shape[1]) 
    input0_2 = Input(shape=o_shape[2]) 
    input1_0 = Input(shape=f_shape[0]) 
    input1_1 = Input(shape=f_shape[1]) 
    input2_0 = Input(shape=f_shape[0]) 
    input2_1 = Input(shape=f_shape[1]) 
    #split0 = Lambda( lambda x: tf.split(x,num_or_size_splits=3,axis=1))(input0)
    #split1 = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=1))(input1)
    #split2 = Lambda( lambda x: tf.split(x,num_or_size_splits=2,axis=1))(input2)
    pl = 'non'
    dp = None
    if mode==18:
        net0_0 = ResNet18(input_tensor=input0_0, pooling=pl,dropout=dp)
        net0_1 = ResNet18(input_tensor=input0_1, pooling=pl,dropout=dp)
        net0_2 = ResNet18(input_tensor=input0_2, pooling=pl,dropout=dp)
        net1_0 = ResNet18(input_tensor=input1_0, pooling=pl,dropout=dp)
        net1_1 = ResNet18(input_tensor=input1_1, pooling=pl,dropout=dp)
        net2_0 = ResNet18(input_tensor=input2_0, pooling=pl,dropout=dp)
        net2_1 = ResNet18(input_tensor=input2_1, pooling=pl,dropout=dp)
    elif mode==34:
        net0_0 = ResNet34(input_tensor=input0_0, pooling=pl,dropout=dp)
        net0_1 = ResNet34(input_tensor=input0_1, pooling=pl,dropout=dp)
        net0_2 = ResNet34(input_tensor=input0_2, pooling=pl,dropout=dp)
        net1_0 = ResNet34(input_tensor=input1_0, pooling=pl,dropout=dp)
        net1_1 = ResNet34(input_tensor=input1_1, pooling=pl,dropout=dp)
        net2_0 = ResNet34(input_tensor=input2_0, pooling=pl,dropout=dp)
        net2_1 = ResNet34(input_tensor=input2_1, pooling=pl,dropout=dp)
    elif mode==50:
        net0_0 = ResNet50(input_tensor=input0_0, pooling=pl,dropout=dp)
        net0_1 = ResNet50(input_tensor=input0_1, pooling=pl,dropout=dp)
        net0_2 = ResNet50(input_tensor=input0_2, pooling=pl,dropout=dp)
        net1_0 = ResNet50(input_tensor=input1_0, pooling=pl,dropout=dp)
        net1_1 = ResNet50(input_tensor=input1_1, pooling=pl,dropout=dp)
        net2_0 = ResNet50(input_tensor=input2_0, pooling=pl,dropout=dp)
        net2_1 = ResNet50(input_tensor=input2_1, pooling=pl,dropout=dp)
    elif mode==101:
        net0_0 = ResNet101(input_tensor=input0_0, pooling=pl,dropout=dp)
        net0_1 = ResNet101(input_tensor=input0_1, pooling=pl,dropout=dp)
        net0_2 = ResNet101(input_tensor=input0_2, pooling=pl,dropout=dp)
        net1_0 = ResNet101(input_tensor=input1_0, pooling=pl,dropout=dp)
        net1_1 = ResNet101(input_tensor=input1_1, pooling=pl,dropout=dp)
        net2_0 = ResNet101(input_tensor=input2_0, pooling=pl,dropout=dp)
        net2_1 = ResNet101(input_tensor=input2_1, pooling=pl,dropout=dp)
    else: #152
        net0_0 = ResNet152(input_tensor=input0_0, pooling=pl,dropout=dp)
        net0_1 = ResNet152(input_tensor=input0_1, pooling=pl,dropout=dp)
        net0_2 = ResNet152(input_tensor=input0_2, pooling=pl,dropout=dp)
        net1_0 = ResNet152(input_tensor=input1_0, pooling=pl,dropout=dp)
        net1_1 = ResNet152(input_tensor=input1_1, pooling=pl,dropout=dp)
        net2_0 = ResNet152(input_tensor=input2_0, pooling=pl,dropout=dp)
        net2_1 = ResNet152(input_tensor=input2_1, pooling=pl,dropout=dp)
    for layer in net0_0.layers:
      layer.name = layer.name + str("_0_0")
      layer.trainable = True
    x0_0 = net0_0.output
    for layer in net0_1.layers:
      layer.name = layer.name + str("_0_1")
      layer.trainable = True
    x0_1 = net0_1.output
    for layer in net0_2.layers:
      layer.name = layer.name + str("_0_2")
      layer.trainable = True
    x0_2 = net0_2.output
    for layer in net1_0.layers:
      layer.name = layer.name + str("_1_0")
      layer.trainable = True
    x1_0 = net1_0.output
    for layer in net1_1.layers:
      layer.name = layer.name + str("_1_1")
      layer.trainable = True
    x1_1 = net1_1.output
    for layer in net2_0.layers:
      layer.name = layer.name + str("_2_0")
      layer.trainable = True
    x2_0 = net2_0.output
    for layer in net2_1.layers:
      layer.name = layer.name + str("_2_1")
      layer.trainable = True
    x2_1 = net2_1.output
    #print (x1.shape)
    x3 = Input((external_dim,))
    x4 = Input((external_dim,))
    x5 = Input((external_dim,))
    
    f = fusion.net_mt5(x0_0,x0_1,x0_2,x1_0,x1_1,x2_0,x2_1,x3,x4,x5)



    #h1 = Dense(units=10)(input3)
    #h1 = Activation('relu')(h1)
    ##h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
    #h2 = Dense(units=2)(h1)
    #activation = Activation('relu')(h1)
   # external_output = Activation('relu')(h2)


    #m4 = add([x1,x2]) 
    #m4 = Dropout(0.5)(m4) 
    #conc = concatenate([external_output, m4])
    #output = Dense(units=10,activation='relu')(conc)
    #f = Dense(units=1,activation='relu')(output)
    
    
    
    model = Model(inputs=[input0_0,input0_1,input0_2,input1_0,input1_1,input2_0,input2_1,x3,x4,x5], outputs=f)   
    
    
    return model



def net_mt5(o_conf=(15, 1, 32, 32),f_conf=(15, 1, 32, 32), external_dim=5,mode=18):#121 169 201 264 161
    lead, f_channels, f_height, f_width = f_conf
    look, o_channels, o_height, o_width = o_conf
    o_shape = (o_channels, look , o_height, o_width)
    f_shape = (f_channels, lead, f_height, f_width)
    pl = 'non'
    dp = None
    input0 = Input(shape=o_shape) 
    input1 = Input(shape=f_shape) 
    input2 = Input(shape=f_shape) 
    #input_f = multiply([input1, input2])
    #input_fusion = concatenate([input0, input_f],axis = 2)
    input_fusion = concatenate([input0, input1, input2],axis = 2)
    print (input_fusion.shape)
    if mode==18:
        res_net0 = ResNet18(input_tensor=input_fusion, pooling=pl,dropout=dp)
        #res_net1 = ResNet18(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==34:
        res_net0 = ResNet34(input_tensor=input_fusion, pooling=pl,dropout=dp)
        #res_net1 = ResNet34(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==50:
        res_net0 = ResNet50(input_tensor=input_fusion, pooling=pl,dropout=dp)
        #res_net1 = ResNet50(input_tensor=input_fusion, pooling=pl,dropout=dp)
    elif mode==101:
        res_net0 = ResNet101(input_tensor=input_fusion, pooling=pl,dropout=dp)
        #res_net1 = ResNet101(input_tensor=input_fusion, pooling=pl,dropout=dp)
    else:
        res_net0 = ResNet152(input_tensor=input_fusion, pooling=pl,dropout=dp)
        #res_net1 = ResNet152(input_tensor=input_fusion, pooling=pl,dropout=dp)
    for layer in res_net0.layers:
        layer.trainable = True
    x0 = res_net0.output
    x3 = Input((external_dim,))
    x4 = Input((external_dim,))
    x5 = Input((external_dim,))
    
    f = fusion.net_mt8(x0,x3,x4,x5)
    
    model = Model(inputs=[input0,input1, input2,x3,x4,x5], outputs=f)   
    
    
    return model
