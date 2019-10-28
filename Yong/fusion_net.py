from keras.layers import *
import keras
import tensorflow as tf


def net(x0,x1,x2,x3):
    dp = 0.50
    #o = Dense(32, activation='relu')(x0) 
    #o = Dense(16,activation='relu')(o)
    o = Dense(10,activation='relu')(x0)
    o = Dropout(dp)(o)   
    o = Dense(1,activation ='relu')(o)   
    m4 = add([x1,x2])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f = Dense(10,activation='relu')(m4)
    f = Dropout(dp)(f)     
    f = Dense(1,activation ='relu')(f)    

    
    #x3 = Dropout(dp)(x3)     
    #q = Dense(32, activation='relu')(x3) 
    #q = Dense(20, activation='relu')(x3) 
    q = Dense(10, activation='relu')(x3) 
    q = Dense(1, activation='relu')(q) 
    out = add([o,q,f])
    return out

def net_mt(x0,x1,x2,x3,x4,x5):
    dp = 0.10
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    print (x0.shape,x1.shape,x2.shape,x3.shape)
    o_1 = Dense(32,activation='relu',name = 'o_dense_1_0')(x0)
    o_1 = Dense(16,activation='relu',name = 'o_dense_1_1')(o_1)
    o_1 = Dropout(dp)(o_1)   
    o_1 = Dense(1,activation ='relu',name = 'o_dense_1_2')(o_1)  
    
    
    o_2 = Dense(32,activation='relu',name = 'o_dense_2_0')(x0)
    o_2 = Dense(16,activation='relu',name = 'o_dense_2_1')(o_2)
    o_2 = Dropout(dp)(o_2)   
    o_2 = Dense(1,activation ='relu',name = 'o_dense_2_2')(o_2) 
    
    
    
    o_3 = Dense(32,activation='relu',name = 'o_dense_3_0')(x0)
    o_3 = Dense(16,activation='relu',name = 'o_dense_3_1')(o_3)
    o_3 = Dropout(dp)(o_3)   
    o_3 = Dense(1,activation ='relu',name = 'o_dense_3_2')(o_3)  
    
    m4 = add([x1,x2])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f) 
    
    
    f_1 = Dense(32,activation='relu',name = 'f_dense_1_0')(m4)
    f_1 = Dense(16,activation='relu',name = 'f_dense_1_1')(f_1)
    f_1 = Dropout(dp)(f_1)     
    f_1 = Dense(1,activation ='relu',name = 'f_dense_1_2')(f_1) 
    
    
    f_2 = Dense(32,activation='relu',name = 'f_dense_2_0')(m4)
    f_2 = Dense(16,activation='relu',name = 'f_dense_2_1')(f_2)
    f_2 = Dropout(dp)(f_2)     
    f_2 = Dense(1,activation ='relu',name = 'f_dense_2_2')(f_2) 
    
    f_3 = Dense(32,activation='relu',name = 'f_dense_3_0')(m4)
    f_3 = Dense(16,activation='relu',name = 'f_dense_3_1')(f_3)
    f_3 = Dropout(dp)(f_3)     
    f_3 = Dense(1,activation ='relu',name = 'f_dense_3_2')(f_3)    

    
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    out1 = add([o_1,q_1,f_1],name = 'out_1')
    out2 = add([o_2,q_2,f_2],name = 'out_2')
    out3 = add([o_3,q_3,f_3],name = 'out_3')
    
    #conc1 = concatenate([o_1,q_1,f_1],name = 'conc_1')
    #out1 = Dense(units=10,activation='relu',name = 'out_1_1')(conc1)
    #out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    #conc2 = concatenate([o_2,q_2,f_2],name = 'conc_2')
    #out2 = Dense(units=10,activation='relu',name = 'out_2_1')(conc2)
    #out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    #conc3 = concatenate([o_3,q_3,f_3],name = 'conc_3')
    #out3 = Dense(units=10,activation='relu',name = 'out_3_1')(conc3)
    #out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    
    return [out1,out2,out3]
def getz(my_init,input ,layername): 
    dp = 0.10   
    of = Conv2D(256, 2, strides=2, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_1')(input)
    print('of shape:'+str(of.shape))
    of = Conv2D(128, 2, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_2')(of)
    of = Conv2D(64, 2, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_3')(of)    
    of = Conv2D(32, 2, activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_4')(of)           
    of = Flatten()(of) 
    of = Dropout(dp)(of)   
    of = Dense(32,activation='relu',name = layername+'dense_1')(of)
    #of = Dense(1,activation ='relu',name = layername+'dense_2')(of) 
    return of
def getz1(my_init,input ,layername): 
    #dp = 0.10   
    #of = Permute((2, 1,3,4))(input)
    #print('of shape:'+str(of.shape))
    of = Conv3D(256, (3,2,2), strides=(2,1,1), activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_1')(input)
    print('conv1 shape:'+str(of.shape))
    of = Conv3D(128, (3,2,2), strides=(2,1,1), activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_2')(of)
    print('conv2 shape:'+str(of.shape))
    of = Conv3D(64, (3,2,2), strides=(2,1,1), activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_3')(of)  
    print('conv3 shape:'+str(of.shape))  
    of = Conv3D(32, (3,2,2), strides=(2,1,1), activation = 'relu', padding = 'same', data_format = 'channels_first',kernel_initializer=my_init,name = layername+'conv_4')(of) 
    print('conv4 shape:'+str(of.shape))
    of = Flatten()(of)  
    of = Dense(32,activation='relu',name = layername+'dense_1')(of)
    
    of = Dense(16,activation ='relu',name = layername+'dense_2')(of) 
    #of = Dropout(dp)(of)  
    #of = Dense(8,activation ='relu',name = layername+'dense_3')(of)
    return of
def net_mt1(x0,x1,x2,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x2.shape,x3.shape)
    m4 = add([x1,x2])
    conc0 = concatenate([x0,m4],name = 'conc_0',axis = 1)
    
    ofG = getz1(my_init,conc0,'ofG')
    ofB = getz1(my_init,conc0,'ofB')
    ofM = getz1(my_init,conc0,'ofM')   
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]

def net_mt2(x0,x1,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    dp = 0.50
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x3.shape)
    conc0 = concatenate([x0,x1],name = 'conc_0',axis = 1)
    
    ofG = getz1(my_init,conc0,'ofG')
    #ofG = Dropout(dp)(ofG)  
    ofB = getz1(my_init,conc0,'ofB')
   #ofB = Dropout(dp)(ofB)  
    ofM = getz1(my_init,conc0,'ofM')   
    #ofM = Dropout(dp)(ofM)  
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    

    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_mt3(x0,x1,x2,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x3.shape)
    m4 = add([x1,x2])
    conc0 = concatenate([x0,m4],name = 'conc_0',axis = 1)
    
    ofG = getz1(my_init,conc0,'ofG')
    ofB = getz1(my_init,conc0,'ofB')
    ofM = getz1(my_init,conc0,'ofM')   
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    #out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(conc1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    #out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(conc2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    #out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(conc3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def final_net(input,layername):    
    out = Dense(units=256,activation='relu',name = layername+'out_1')(input)
    out = Dense(units=128,activation='relu',name = layername+'out_2')(out)
    out = Dense(units=64,activation='relu',name = layername+'out_3')(out)
    out = Dense(units=32,activation='relu',name = layername+'out_4')(out)
    out = Dense(units=16,activation='relu',name = layername+'out_5')(out)
    out = Dense(units=1,activation='relu',name = layername+'out_6')(out)
    return out
def net_mt3_1(x0,x1,x2,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x3.shape)
    #m4 = add([x1,x2])
    conc0 = concatenate([x0,x1,x2],name = 'conc_0',axis = 1)
    
    ofG = getz1(my_init,conc0,'ofG')
    ofB = getz1(my_init,conc0,'ofB')
    ofM = getz1(my_init,conc0,'ofM')   
    
    #print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(64, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(64, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(64, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    #out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    #out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    #out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    out1 = final_net(conc1,'G_out')
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = final_net(conc2,'B_out')
    #out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    #out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    #out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = final_net(conc3,'M_out')
    #out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    #out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    #out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_mt6(x0,x1,x2,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x3.shape)
    m4 = subtract([x1,x2])
    conc0 = concatenate([x0,m4],name = 'conc_0',axis = 2)
    
    ofG = getz1(my_init,conc0,'ofG')
    ofB = getz1(my_init,conc0,'ofB')
    ofM = getz1(my_init,conc0,'ofM')   
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_mt7(x0,x1,x2,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
   
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    print (x0.shape,x1.shape,x3.shape)
    m4 = multiply([x1,x2])
    concG0 = concatenate([x0,m4],name = 'concG_0',axis = 2)
    m5 = multiply([x1,x2])
    concB0 = concatenate([x0,m5],name = 'concB_0',axis = 2)
    m6 = subtract([x1,x2])
    #m6 = Lambda(lambda inputs: inputs[0] / inputs[1])([x1,x2])
    concM0 = concatenate([x0,m6],name = 'concM_0',axis = 2)
    
    ofG = getz1(my_init,concG0,'ofG')
    ofB = getz1(my_init,concB0,'ofB')
    ofM = getz1(my_init,concM0,'ofM')        
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_mt8(x0,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    
    ofG = getz1(my_init,x0,'ofG')
    ofB = getz1(my_init,x0,'ofB')
    ofM = getz1(my_init,x0,'ofM')  
    
    
    #ofG = getz1(my_init,x0,'ofG')
    #ofB = getz1(my_init,x0,'ofB')
    #ofM = getz1(my_init,x0,'ofM')   
    
    #print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(10, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(10, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(10, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    

    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_v0(x0,x1,x2,x3):
    dp = 0.50
    print (x0.shape,x1.shape,x2.shape,x3.shape)
    #o = Dense(32, activation='relu')(x0) 
    #o = Dense(16,activation='relu')(o)
    o = Dense(10,activation='relu')(x0)
    o = Dropout(dp)(o)   
    o = Dense(1,activation ='relu')(o)   
    m4 = add([x1,x2])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f = Dense(10,activation='relu')(m4)
    f = Dropout(dp)(f)     
    f = Dense(1,activation ='relu')(f)    

    
    #x3 = Dropout(dp)(x3)     
    #q = Dense(32, activation='relu')(x3) 
    #q = Dense(16, activation='relu')(q) 
    q = Dense(10, activation='relu')(x3) 
    q = Dense(1, activation='relu')(q) 
    out = add([o,q,f])
    return q,o,f,out
def net_v1(x0,x1,x2,x3):
    dp = 0.50
    #x3 = Dropout(dp)(x3)     
    #q = Dense(32, activation='relu')(x3) 
    #q = Dense(16, activation='relu')(q) 
    q = Dense(10, activation='relu')(x3) 
    q = Dense(1, activation='relu')(q) 
    
    #o = Dense(32, activation='relu')(x0) 
    #o = Dense(16,activation='relu')(o)
    o = Dense(10,activation='relu')(x0)
    o = Dropout(dp)(o)   
    go1 =  concatenate([o,x3]) 
    o = Dense(1,activation ='relu')(o) 
    go1 = Dense(1, activation='sigmoid')(go1)
    o = multiply([o, go1])
    
    
    m4 = add([x1,x2])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f = Dense(10,activation='relu')(m4)
    f = Dropout(dp)(f)  
    go2 =  concatenate([f,x3])   
    f = Dense(1,activation ='relu')(f) 
    go2 = Dense(1, activation='sigmoid')(go2)  
    f = multiply([f, go2]) 

    
    f = add([o,q,f])
    return f
    
def net_v2(x0,x1,x2):
    dp = 0.60
    #o = Dense(32, activation='relu')(x0) 
    #o = Dense(16,activation='relu')(o)
    o = Dense(10,activation='relu')(x0)
    o = Dropout(dp)(o)   
    o = Dense(1,activation ='relu')(o)   
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f = Dense(10,activation='relu')(x1)
    f = Dropout(dp)(f)     
    f = Dense(1,activation ='relu')(f)    

    
    #x3 = Dropout(dp)(x3)     
    #q = Dense(32, activation='relu')(x3) 
    #q = Dense(16, activation='relu')(q) 
    q = Dense(10, activation='relu')(x2) 
    q = Dense(1, activation='relu')(q) 
    f = add([o,q,f])


    #h1 = Dense(units=10)(x3)
    #h1 = Activation('relu')(h1)
    ##h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
    #h2 = Dense(units=2)(h1)
    #h2 = Activation('relu')(h2)


    ##o = Dense(10,activation='relu')(x0)
    #o = Dropout(dp)(x0)       
    #m4 = add([x1,x2]) 
    ##m4 = Dense(10,activation='relu')(m4)
    #m4 = Dropout(dp)(m4) 
    #conc = concatenate([h2,o, m4])
    #output = Dense(units=10,activation='relu')(conc)
    ##output = Dropout(dp)(output)
    #f = Dense(units=1,activation='relu')(output)
    
    
      
    
    
    return f



    
def net_v3(x0,x1,x2):
    dp = 0.50  
    of = concatenate([x0,x1])
    of = Dense(10,activation='relu')(of)
    of = Dropout(dp)(of)   
    
    
    #q = Dense(10, activation='relu')(x2) 
    #q = Dense(1, activation='relu')(q) 
    
    q = concatenate([x2,of])
    q = Reshape((-1,1))(q)
    print (q.shape)
#model.add(Reshape((6, 2)))
    f = LSTM(32, return_sequences=True, dropout=dp, name ='lstm1')(q)
    f = LSTM(16, return_sequences=True, dropout=dp, name ='lstm2')(f)
    f = Flatten()(f)
    f = Dense(units=1,activation='relu')(f)

    #h1 = Dense(units=10)(x3)
    #h1 = Activation('relu')(h1)
    ##h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
    #h2 = Dense(units=2)(h1)
    #h2 = Activation('relu')(h2)


    ##o = Dense(10,activation='relu')(x0)
    #o = Dropout(dp)(x0)       
    #m4 = add([x1,x2]) 
    ##m4 = Dense(10,activation='relu')(m4)
    #m4 = Dropout(dp)(m4) 
    #conc = concatenate([h2,o, m4])
    #output = Dense(units=10,activation='relu')(conc)
    ##output = Dropout(dp)(output)
    #f = Dense(units=1,activation='relu')(output)
    
    
      
    
    
    return f

    
def net_v4(x0,x1,x2,x3):
    dp = 0.50 
    x1 = add([x1,x2])
    of = concatenate([x0,x1])
    of = Dense(10,activation='relu')(of)
    of = Dropout(dp)(of)   
    
    
    #q = Dense(10, activation='relu')(x2) 
    #q = Dense(1, activation='relu')(q) 
    
    q = concatenate([x3,of])
    q = Reshape((-1,1))(q)
    print (q.shape)
#model.add(Reshape((6, 2)))
    f = LSTM(32, return_sequences=True, dropout=dp, name ='lstm1')(q)
    f = LSTM(16, return_sequences=True, dropout=dp, name ='lstm2')(f)
    f = Flatten()(f)
    f = Dense(units=1,activation='relu')(f)

    #h1 = Dense(units=10)(x3)
    #h1 = Activation('relu')(h1)
    ##h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
    #h2 = Dense(units=2)(h1)
    #h2 = Activation('relu')(h2)


    ##o = Dense(10,activation='relu')(x0)
    #o = Dropout(dp)(x0)       
    #m4 = add([x1,x2]) 
    ##m4 = Dense(10,activation='relu')(m4)
    #m4 = Dropout(dp)(m4) 
    #conc = concatenate([h2,o, m4])
    #output = Dense(units=10,activation='relu')(conc)
    ##output = Dropout(dp)(output)
    #f = Dense(units=1,activation='relu')(output)
    
    
      
    
    
    return f



def net_v5(x0_0,x0_1,x0_2,x1_0,x1_1,x2_0,x2_1,x3):
    dp = 0.50
    print (x0_0.shape,x0_1.shape,x0_2.shape,x1_1.shape)
    #o = Dense(32, activation='relu')(x0) 
    #o = Dense(16,activation='relu')(o)
    o_0 = Dense(10,activation='relu')(x0_0)
    o_0 = Dropout(dp)(o_0)   
    o_0 = Dense(1,activation ='relu')(o_0) 
    o_1 = Dense(10,activation='relu')(x0_1)
    o_1 = Dropout(dp)(o_1)   
    o_1 = Dense(1,activation ='relu')(o_1)   
    o_2 = Dense(10,activation='relu')(x0_2)
    o_2 = Dropout(dp)(o_2)   
    o_2 = Dense(1,activation ='relu')(o_2)     
    m4_0 = add([x1_0,x2_0])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f_0 = Dense(10,activation='relu')(m4_0)
    f_0 = Dropout(dp)(f_0)     
    f_0 = Dense(1,activation ='relu')(f_0)     
    m4_1 = add([x1_1,x2_1])
    #f = Dense(32, activation='relu')(m4)   
    #f = Dense(16,activation='relu')(f)  
    f_1 = Dense(10,activation='relu')(m4_1)
    f_1 = Dropout(dp)(f_1)     
    f_1 = Dense(1,activation ='relu')(f_1)       

    
    #x3 = Dropout(dp)(x3)     
    #q = Dense(32, activation='relu')(x3) 
    #q = Dense(16, activation='relu')(q) 
    q = Dense(10, activation='relu')(x3) 
    q = Dense(1, activation='relu')(q) 
    f = add([o_0,o_1,o_2,f_0,f_1,q])
    return f

def net_mt4(x0_0,x0_1,x0_2,x1_0,x1_1,x2_0,x2_1,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    #print (x0.shape,x1.shape,x3.shape)
    #conc0 = concatenate([x0,m4],name = 'conc_0',axis = 1)
    Gf_0 = add([x1_0,x2_0])
    Gf_0 = getz1(my_init,Gf_0,'Gf_0')
    Gf_1 = add([x1_1,x2_1])
    Gf_1 = getz1(my_init,Gf_1,'Gf_1')
    
    Go_0 = getz1(my_init,x0_0,'Go_0')
    Go_1 = getz1(my_init,x0_1,'Go_1')
    Go_2 = getz1(my_init,x0_2,'Go_2')
    
    ofG = concatenate([Go_0,Go_1,Go_2,Gf_0,Gf_1],name = 'ofG',axis = 1)    
    ofG = Dense(32, activation='relu', name='ofG1')(ofG) 
    
    Bf_0 = add([x1_0,x2_0])
    Bf_0 = getz1(my_init,Bf_0,'Bf_0')
    Bf_1 = add([x1_1,x2_1])
    Bf_1 = getz1(my_init,Bf_1,'Bf_1')
    
    Bo_0 = getz1(my_init,x0_0,'Bo_0')
    Bo_1 = getz1(my_init,x0_1,'Bo_1')
    Bo_2 = getz1(my_init,x0_2,'Bo_2')
    
    ofB = concatenate([Bo_0,Bo_1,Bo_2,Bf_0,Bf_1],name = 'ofB',axis = 1)  
    ofB = Dense(16, activation='relu', name='ofB1')(ofB) 
    
    
    Mf_0 = add([x1_0,x2_0])
    Mf_0 = getz1(my_init,Mf_0,'Mf_0')
    Mf_1 = add([x1_1,x2_1])
    Mf_1 = getz1(my_init,Mf_1,'Mf_1')
    
    Mo_0 = getz1(my_init,x0_0,'Mo_0')
    Mo_1 = getz1(my_init,x0_1,'Mo_1')
    Mo_2 = getz1(my_init,x0_2,'Mo_2')
    
    ofM = concatenate([Mo_0,Mo_1,Mo_2,Mf_0,Mf_1],name = 'ofM',axis = 1)  
    ofM = Dense(8, activation='relu', name='ofM1')(ofM) 
    
    #ofG = getz1(my_init,conc0,'ofG')
    #ofB = getz1(my_init,conc0,'ofB')
    #ofM = getz1(my_init,conc0,'ofM')   
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    
    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=32,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=16,activation='relu',name = 'out_1_1')(out1)
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofB],name = 'conc_2')
    out2 = Dense(units=32,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=16,activation='relu',name = 'out_2_1')(out2)
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofM],name = 'conc_3')
    out3 = Dense(units=32,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=16,activation='relu',name = 'out_3_1')(out3)
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
def net_mt5(x0_0,x0_1,x0_2,x1_0,x1_1,x2_0,x2_1,x3,x4,x5):
    #https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
    dp=10
    my_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=9999)
    #print (x0.shape,x1.shape,x3.shape)
    #conc0 = concatenate([x0,m4],name = 'conc_0',axis = 1)
    Gf_0 = add([x1_0,x2_0])
    Gf_0 = getz1(my_init,Gf_0,'Gf_0')
    Gf_1 = add([x1_1,x2_1])
    Gf_1 = getz1(my_init,Gf_1,'Gf_1')
    
    Go_0 = getz1(my_init,x0_0,'Go_0')
    Go_1 = getz1(my_init,x0_1,'Go_1')
    Go_2 = getz1(my_init,x0_2,'Go_2')
    
    ofG = concatenate([Go_0,Go_1,Go_2,Gf_0,Gf_1],name = 'ofG',axis = 1)    
    ofG = Dense(16, activation='relu', name='ofG1')(ofG)     
    
    #ofG = getz1(my_init,conc0,'ofG')
    #ofB = getz1(my_init,conc0,'ofB')
    #ofM = getz1(my_init,conc0,'ofM')   
    
    print ('ofG shape '+str(ofG.shape))  
    #x3 = Dropout(dp)(x3)     
    q_1 = Dense(32, activation='relu',name = 'q_dense_1_0')(x3)  
    #go1 = Dense(1, activation='sigmoid')(x3)
    q_1 = Dense(16, activation='relu',name = 'q_dense_1_1')(q_1) 
    #q_1 = Dense(1, activation='relu',name = 'q_dense_1_2')(q_1) 
    
    q_2 = Dense(32, activation='relu',name = 'q_dense_2_0')(x4) 
    #go2 = Dense(1, activation='sigmoid')(x4)
    q_2 = Dense(16, activation='relu',name = 'q_dense_2_1')(q_2) 
    #q_2 = Dense(1, activation='relu',name = 'q_dense_2_2')(q_2) 
    
    q_3 = Dense(32, activation='relu',name = 'q_dense_3_0')(x5) 
    #go3 = Dense(1, activation='sigmoid')(x5)
    q_3 = Dense(16, activation='relu',name = 'q_dense_3_1')(q_3) 
    #q_3 = Dense(1, activation='relu',name = 'q_dense_3_2')(q_3) 
    
    
    #o_1 = multiply([o_1, go1])
    #o_2 = multiply([o_2, go2])
    #o_3 = multiply([o_3, go3])
    
    #f_1 = multiply([f_1, go1])
    #f_2 = multiply([f_2, go2])
    #f_3 = multiply([f_3, go3])
    
    #out1 = add([q_1,ofG],name = 'out_1')
    #out2 = add([q_2,ofB],name = 'out_2')
    #out3 = add([q_3,ofM],name = 'out_3')
    #ofG = multiply([ofG,go1])
    #ofB = multiply([ofG,go2])
    #ofM = multiply([ofG,go3])


    conc1 = concatenate([q_1,ofG],name = 'conc_1')
    out1 = Dense(units=16,activation='relu',name = 'out_1_0')(conc1)
    out1 = Dense(units=10,activation='relu',name = 'out_1_1')(out1)
    #out1 = Dropout(dp)(out1)  
    out1 = Dense(units=1,activation='relu',name = 'out_1_2')(out1)
    conc2 = concatenate([q_2,ofG],name = 'conc_2')
    out2 = Dense(units=16,activation='relu',name = 'out_2_0')(conc2)
    out2 = Dense(units=10,activation='relu',name = 'out_2_1')(out2)
    #out2 = Dropout(dp)(out2)  
    out2 = Dense(units=1,activation='relu',name = 'out_2_2')(out2)
    conc3 = concatenate([q_3,ofG],name = 'conc_3')
    out3 = Dense(units=16,activation='relu',name = 'out_3_0')(conc3)
    out3 = Dense(units=10,activation='relu',name = 'out_3_1')(out3)
    #out3 = Dropout(dp)(out3)  
    out3 = Dense(units=1,activation='relu',name = 'out_3_2')(out3)
    return [out1,out2,out3]
