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

    def __init__(self, batch_size,epochs,X_train, X_test, y_train, y_test):
        self.MODEL_FOLDER = "./model"
        self.batch_size=batch_size
        self.epochs=epochs
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def castle(self,observed_conf=(15, 689), forecasted_conf = (10,689),latent_dim = 256,batchNormalization=False, regularization=False): 
        look, dim = observed_conf
        lead, _ = forecasted_conf
        input_encoder_streamflow = Input(shape=(look, 1),name="look_forward_stream_flow_input")
        input_decoder_streamflow = Input(shape=(lead, 1),name="leadtime_stream_flow_input")
        input_observed = Input(shape=rf_look_conf,name="observed_rainfall_input")
        input_forecasted = Input(shape=rf_lead_conf,name="forecasted_rainfall_input")
        hidden_observed_rainfall = TimeDistributed(Dense(1, activation='relu'), name="h_ob")(input_observed)
        hidden_observed_rainfall = TimeDistributed(concatenate([rf, input_streamflow]), name="conc_h_sf")

        encoder = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.0, name ='lstm_look')
        encoder_outputs, state_h, state_c = encoder(hidden_observed_rainfall)
        encoder_states = [state_h, state_c]
        pred_ob_sf = TimeDistributed(Dense(1, activation='relu'), name="out_ob_sf")(encoder_outputs)

        hidden_forecasted_rainfall = TimeDistributed(Dense(1, activation='relu'), name="h_fo")(input_forecasted)
        hidden_forecasted_rainfall = TimeDistributed(concatenate([rf, pred_ob_sf[-1]]), name="conc_h_sf")

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.0, name ='lstm_look')
        decoder_outputs, _, _ = decoder_lstm(hidden_forecasted_rainfall,
                                             initial_state=encoder_states)
        pred_fo_sf = TimeDistributed(Dense(1, activation='relu'), name="out_fo_sf")(decoder_outputs)

        self.model = Model([input_streamflow, input_observed,input_forecasted], [pred_ob_sf,pred_fo_sf])

        self.model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        
        self.encoder_model = Model([input_streamflow, input_observed], encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def fit(self,X,y)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4) 
        callbacks = [early_stopping,reduce_lr]
        history = self.model.fit(self.X_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                    validation_split = 0.2,
                    #validation_data = (X_test, y_test),
                    callbacks=callbacks,
                    verbose=2)
        clf.save(os.path.join(MODEL_FOLDER, f"castle.h5"))
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
    
    def encoder_decoder_model(self):
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
 

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models




def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence








import numpy as np
import pylab as plt

def convLSTM_net(rf_look_conf=(15, 689), rf_lead_conf = (10,689), external_dim=8,  kernel_size=(3, 3), filters=40, nb_stack=1, batchNormalization=False, regularization=True): 
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)




    look, dim = rf_look_conf
    lead, _ = rf_look_conf
    input_sf = Input(shape=(look, 1),name="stream_flow_input")
    input_rf_look = Input(shape=rf_look_conf,name="rainfall_look_input")
    input_rf_lead = Input(shape=rf_lead_conf,name="rainfall_lead_input")
    rf = TimeDistributed(Dense(1, activation='relu'))(input_rf_look)
    rf = TimeDistributed(concatenate([rf, input_sf]), name="conc_1")
    
    lstm = LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.0, name ='lstm_look')(rf)
    pr = TimeDistributed(concatenate([rf, input_sf]), name="conc_1")
    
    for l in lead:
        
    
    
    input_fe = Input((features_dim),name="claim_features")
    
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
