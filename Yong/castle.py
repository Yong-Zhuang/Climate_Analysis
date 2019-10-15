"""
Created on Sat October 13 21:04:06 2019

@author: Yong Zhuang
"""
import os
import numpy as np
import keras
from keras import regularizers
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
    TimeDistributed,
    LSTM,
    Dropout,
)
from keras.callbacks import EarlyStopping,  Callback, ReduceLROnPlateau
from keras.models import Model
from keras import optimizers

class CASTLE:

    def __init__(self, batch_size,epochs,observed_rf_conf=(15, 689), forecasted_rf_conf = (10,689),sf_dim = 5,latent_dim = 256,batchNormalization=False, regularization=False):
        self.MODEL_FOLDER = "./MODEL"
        self.batch_size=batch_size
        self.epochs=epochs
        self.sf_dim = sf_dim
        
        look, dim_ob = observed_rf_conf
        lead, dim_fo = forecasted_rf_conf
        input_encoder_streamflow = Input(shape=(None, sf_dim),name="look_forward_stream_flow_input")
        input_decoder_streamflow = Input(shape=(None, sf_dim),name="leadtime_stream_flow_input")
        input_observed = Input(shape=(None,dim_ob),name="observed_rainfall_input")
        input_forecasted = Input(shape=(None,dim_fo),name="forecasted_rainfall_input")
        
        #encoder......
        encoder_rf_dense = Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01))
        hidden_observed_rainfall = TimeDistributed(encoder_rf_dense, name="h_ob1")(input_observed)
        
        hidden_observed_rainfall = Dropout(0.5)(hidden_observed_rainfall)        
       # hidden_observed_rainfall = BatchNormalization()(hidden_observed_rainfall)
        
       # encoder_input_dense2 = Dense(1000, activation='relu',
       #         kernel_regularizer=regularizers.l2(0.01),
       #         activity_regularizer=regularizers.l2(0.01),
       #         bias_regularizer=regularizers.l2(0.01))
       # hidden_observed_rainfall = TimeDistributed(encoder_input_dense2, name="h_ob2")(hidden_observed_rainfall)
       # 
       # hidden_observed_rainfall = Dropout(0.5)(hidden_observed_rainfall)        
       # hidden_observed_rainfall = BatchNormalization()(hidden_observed_rainfall)
        
        encoder_sf_dense = Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01))        
        hidden_sf = TimeDistributed(encoder_sf_dense, name="h_sf")(input_encoder_streamflow)
        hidden_sf = Dropout(0.5)(hidden_sf)  
        
        hidden_observed_rainfall = concatenate([hidden_observed_rainfall,hidden_sf],axis = 2, name = "conc_h_look")
        hidden_observed_rainfall = BatchNormalization()(hidden_observed_rainfall)
        encoder = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0, recurrent_dropout=0, name ='lstm_look')
        encoder_outputs, state_h, state_c = encoder(hidden_observed_rainfall)
        encoder_states = [state_h, state_c]
        pred_ob_sf = TimeDistributed(Dense(1, activation='relu'), name="out_ob_sf")(encoder_outputs)

        #decoder......
#         decoder_rf_dense = Dense(512, activation='relu',
#                         kernel_regularizer=regularizers.l2(0.01),
#                         activity_regularizer=regularizers.l2(0.01),
#                         bias_regularizer=regularizers.l2(0.01))
        hidden_forecasted_rainfall = TimeDistributed(encoder_rf_dense, name="h_fo1")(input_forecasted)
        
        hidden_forecasted_rainfall = Dropout(0.5)(hidden_forecasted_rainfall)
        #hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)
        
        #decoder_input_dense2 = Dense(1000, activation='relu',
        #                kernel_regularizer=regularizers.l2(0.01),
        #                activity_regularizer=regularizers.l2(0.01),
        #                bias_regularizer=regularizers.l2(0.01))
        #hidden_forecasted_rainfall = TimeDistributed(decoder_input_dense2, name="h_fo2")(hidden_forecasted_rainfall)
        
        #hidden_forecasted_rainfall = Dropout(0.5)(hidden_forecasted_rainfall)
        #hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)
               
        hidden_sf2 = TimeDistributed(encoder_sf_dense, name="h_sf2")(input_decoder_streamflow)
        hidden_sf2 = Dropout(0.5)(hidden_sf2) 
        hidden_forecasted_rainfall = concatenate([hidden_forecasted_rainfall,hidden_sf2],axis = 2, name = 'conc_h_lead')
        hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)
        #decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0, recurrent_dropout=0, name ='lstm_lead')
        decoder_lstm = encoder
        decoder_outputs, _, _ = decoder_lstm(hidden_forecasted_rainfall,initial_state=encoder_states)
        decoder_dense = TimeDistributed(Dense(1, activation='relu'), name="out_fo_sf")
        pred_fo_sf = decoder_dense(decoder_outputs)
       
        self.model = Model([input_encoder_streamflow, input_decoder_streamflow, input_observed,input_forecasted], [pred_ob_sf,pred_fo_sf])
        adam = optimizers.Adam(lr=0.1)
        self.model.compile(loss="mse", optimizer=adam, metrics=["mae"])
        print ("pred_ob_sf.shape is "+str(pred_ob_sf.shape))
        self.encoder_model = Model([input_encoder_streamflow, input_observed], [pred_ob_sf]+encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            hidden_forecasted_rainfall, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [input_decoder_streamflow, input_forecasted] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)        

    def fit(self,X,y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4) 
        callbacks = [early_stopping,reduce_lr]
        history = self.model.fit(X, y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                    validation_split = 0.2,
                    #validation_data = (X_test, y_test),
                    callbacks=callbacks,
                    verbose=2)
        self.model.save(os.path.join(self.MODEL_FOLDER, "castle.h5"))
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
    def predict(self,input_encoder_streamflow, input_observed,input_forecasted):
        # Encode the input as state vectors.
        
        init_prediction, state_h, state_c = self.encoder_model.predict([input_encoder_streamflow,input_observed])
        prediction = init_prediction[:,-self.sf_dim:]
        print(init_prediction.shape, prediction.shape)
        input_decoder_streamflow = prediction
        # Sampling loop for a batch of sequences
        
        for i in range(input_forecasted.shape[1]):
            output, h, c = self.decoder_model.predict(
                [input_decoder_streamflow,input_forecasted[:,i:i+1]] + [state_h,state_c])
	    #print("output shape is "+str(output.shape))
            input_decoder_streamflow = np.append(input_decoder_streamflow, output[:,-1:],axis = 1)
            input_decoder_streamflow = input_decoder_streamflow[:,1:]
	    #print (prediction.shape, input_decoder_streamflow.shape)
            prediction = np.append(prediction,input_decoder_streamflow, axis = 1)
	
            # Update states
            states_value = [h, c]
        print(prediction.shape)
        return prediction
