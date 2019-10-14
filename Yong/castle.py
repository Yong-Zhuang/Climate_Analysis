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

class CASTLE

    def __init__(self, batch_size,epochs,observed_conf=(15, 689), forecasted_conf = (10,689),latent_dim = 256,batchNormalization=False, regularization=False):
        self.MODEL_FOLDER = "./model"
        self.batch_size=batch_size
        self.epochs=epochs
        
        look, dim = observed_conf
        lead, _ = forecasted_conf
        input_encoder_streamflow = Input(shape=(look, 1),name="look_forward_stream_flow_input")
        input_decoder_streamflow = Input(shape=(lead, 1),name="leadtime_stream_flow_input")
        input_observed = Input(shape=rf_look_conf,name="observed_rainfall_input")
        input_forecasted = Input(shape=rf_lead_conf,name="forecasted_rainfall_input")
        hidden_observed_rainfall = TimeDistributed(Dense(1, activation='relu'), name="h_ob")(input_observed)
        hidden_observed_rainfall = TimeDistributed(concatenate([hidden_observed_rainfall, input_encoder_streamflow]), name="conc_h_sf")

        encoder = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.0, name ='lstm_look')
        encoder_outputs, state_h, state_c = encoder(hidden_observed_rainfall)
        encoder_states = [state_h, state_c]
        pred_ob_sf = TimeDistributed(Dense(1, activation='relu'), name="out_ob_sf")(encoder_outputs)

        hidden_forecasted_rainfall = TimeDistributed(Dense(1, activation='relu'), name="h_fo")(input_forecasted)
        hidden_forecasted_rainfall = TimeDistributed(concatenate([hidden_forecasted_rainfall, input_decoder_streamflow]), name="conc_h_sf")

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.0, name ='lstm_look')
        decoder_outputs, _, _ = decoder_lstm(hidden_forecasted_rainfall,initial_state=encoder_states)
        pred_fo_sf = TimeDistributed(Dense(1, activation='relu'), name="out_fo_sf")(decoder_outputs)

        self.model = Model([input_encoder_streamflow, input_decoder_streamflow, input_observed,input_forecasted], [pred_ob_sf,pred_fo_sf])

        self.model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        print (f"pred_ob_sf.shape is {pred_ob_sf.shape}")
        self.encoder_model = Model([input_encoder_streamflow, input_observed], [pred_ob_sf[-1],encoder_states])

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [input_decoder_streamflow, input_forecasted] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)        

    def fit(self,X,y)
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
        clf.save(os.path.join(self.MODEL_FOLDER, f"castle.h5"))
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
    def predict(self,input_encoder_streamflow, input_observed,input_forecasted):
        # Encode the input as state vectors.
        
        init_prediction, states_value = self.encoder_model.predict([input_encoder_streamflow,input_observed])
        prediction = init_prediction
        input_decoder_streamflow = init_prediction
        # Sampling loop for a batch of sequences
        for i in range(input_forecasted.shape[1]):
            output, h, c = decoder_model.predict(
                [init_prediction,input_forecasted[:,i]] + states_value)
            input_decoder_streamflow = output[-1]
            prediction = np.append(prediction,input_decoder_streamflow, axis = 1)
            # Update states
            states_value = [h, c]

        return prediction