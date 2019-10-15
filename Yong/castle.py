"""
Created on Sat October 13 21:04:06 2019

@author: Yong Zhuang
"""
import os
import numpy as np
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Flatten, concatenate, TimeDistributed, GRU, Dropout, Conv1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from layers.attention import AttentionLayer


class CASTLE:
    def __init__(
        self, batch_size, epochs, observed_rf_conf=(15, 689), forecasted_rf_conf=(10, 689), sf_dim=5, latent_dim=256
    ):
        self.MODEL_FOLDER = "./MODEL"
        self.batch_size = batch_size
        self.epochs = epochs
        self.sf_dim = sf_dim

        look, dim_ob = observed_rf_conf
        lead, dim_fo = forecasted_rf_conf
        input_encoder_streamflow = Input(shape=(None, sf_dim, 1), name="look_forward_stream_flow_input")
        input_decoder_streamflow = Input(shape=(None, sf_dim, 1), name="leadtime_stream_flow_input")
        input_encoder_observed_rf = Input(shape=(None, dim_ob), name="observed_rainfall_input")
        input_decoder_forecasted_rf = Input(shape=(None, dim_fo), name="forecasted_rainfall_input")

        # encoder......
        encoder_rf_dense = Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
        )
        hidden_observed_rainfall = TimeDistributed(encoder_rf_dense, name="h_ob1")(input_encoder_observed_rf)

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

        #         encoder_sf_dense = Dense(512, activation='relu',
        #                         kernel_regularizer=regularizers.l2(0.01),
        #                         activity_regularizer=regularizers.l2(0.01),
        #                         bias_regularizer=regularizers.l2(0.01))
        encoder_sf_cnn1 = Conv1D(
            filters=512,
            kernel_size=2,
            strides=1,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
            padding="same",
        )
        encoder_sf_cnn2 = Conv1D(
            filters=256,
            kernel_size=3,
            strides=1,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
            padding="same",
        )
        encoder_sf_cnn3 = Conv1D(
            filters=256,
            kernel_size=5,
            strides=2,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
            padding="same",
        )
        encoder_sf_cnn4 = Conv1D(
            filters=256,
            kernel_size=7,
            strides=3,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
            padding="same",
        )
        # input_encoder_sf = K.expand_dims(input_encoder_streamflow,axis=-1)
        # input_encoder_sf = Reshape((None, sf_dim,1))(input_encoder_streamflow)
        # print(input_encoder_streamflow.shape)

        hidden_sf = TimeDistributed(encoder_sf_cnn1, name="h_sf1")(input_encoder_streamflow)
        hidden_sf = TimeDistributed(encoder_sf_cnn2, name="h_sf2")(hidden_sf)
        hidden_sf = TimeDistributed(encoder_sf_cnn3, name="h_sf3")(hidden_sf)
        hidden_sf = TimeDistributed(encoder_sf_cnn4, name="h_sf0")(hidden_sf)
        hidden_sf = TimeDistributed(Flatten())(hidden_sf)
        hidden_sf = Dropout(0.5)(hidden_sf)

        hidden_observed_rainfall = concatenate([hidden_observed_rainfall, hidden_sf], axis=2, name="conc_h_look")
        hidden_observed_rainfall = BatchNormalization()(hidden_observed_rainfall)
        encoder = GRU(
            latent_dim, return_state=True, return_sequences=True, dropout=0, recurrent_dropout=0, name="lstm_look"
        )
        encoder_outputs, encoder_states = encoder(hidden_observed_rainfall)
        encoder_pred = TimeDistributed(Dense(1, activation="relu"), name="out_ob_sf")(encoder_outputs)

        # decoder......
        # decoder_rf_dense = Dense(512, activation='relu',
        #                kernel_regularizer=regularizers.l2(0.01),
        #                activity_regularizer=regularizers.l2(0.01),
        #                bias_regularizer=regularizers.l2(0.01))
        hidden_forecasted_rainfall = TimeDistributed(encoder_rf_dense, name="h_fo1")(input_decoder_forecasted_rf)

        hidden_forecasted_rainfall = Dropout(0.5)(hidden_forecasted_rainfall)
        # hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)

        # decoder_input_dense2 = Dense(1000, activation='relu',
        #                kernel_regularizer=regularizers.l2(0.01),
        #                activity_regularizer=regularizers.l2(0.01),
        #                bias_regularizer=regularizers.l2(0.01))
        # hidden_forecasted_rainfall = TimeDistributed(decoder_input_dense2, name="h_fo2")(hidden_forecasted_rainfall)

        # hidden_forecasted_rainfall = Dropout(0.5)(hidden_forecasted_rainfall)
        # hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)

        # hidden_sf2 = TimeDistributed(encoder_sf_dense, name="h_sf2")(input_decoder_streamflow)
        # input_decoder_sf = K.expand_dims(input_decoder_streamflow,axis=-1)
        # input_decoder_sf = Reshape((None, sf_dim,1))(input_decoder_streamflow)
        hidden_sf2 = TimeDistributed(encoder_sf_cnn1, name="h_sf4")(input_decoder_streamflow)
        hidden_sf2 = TimeDistributed(encoder_sf_cnn2, name="h_sf5")(hidden_sf2)
        hidden_sf2 = TimeDistributed(encoder_sf_cnn3, name="h_sf6")(hidden_sf2)
        hidden_sf2 = TimeDistributed(encoder_sf_cnn4, name="h_sf7")(hidden_sf2)
        hidden_sf2 = TimeDistributed(Flatten())(hidden_sf2)
        hidden_sf2 = Dropout(0.5)(hidden_sf2)
        hidden_forecasted_rainfall = concatenate([hidden_forecasted_rainfall, hidden_sf2], axis=2, name="conc_h_lead")
        hidden_forecasted_rainfall = BatchNormalization()(hidden_forecasted_rainfall)
        # decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0,
        #                     recurrent_dropout=0, name ='lstm_lead')
        decoder_gru = encoder
        decoder_outputs, decoder_states = decoder_gru(hidden_forecasted_rainfall, initial_state=encoder_states)

        # Attention layer
        attn_layer = AttentionLayer(name="attention_layer")
        attn_outputs, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder GRU output
        decoder_concat_input = concatenate(axis=-1, name="concat_layer")([decoder_outputs, attn_outputs])

        # Dense layer
        decoder_dense_time = TimeDistributed(Dense(1, activation="relu"), name="decoder_time_distributed_layer")
        decoder_pred = decoder_dense_time(decoder_concat_input)

        # Full model
        self.full_model = Model(
            [
                input_encoder_streamflow,
                input_decoder_streamflow,
                input_encoder_observed_rf,
                input_decoder_forecasted_rf,
            ],
            [encoder_pred, decoder_pred],
        )
        adam = optimizers.Adam(lr=0.1)
        self.model.compile(loss="mse", optimizer=adam, metrics=["mae"])
        print("decoder_pred.shape is " + str(decoder_pred.shape))

        """ Inference model """
        self.encoder_model = Model(
            [input_encoder_streamflow, input_encoder_observed_rf], [encoder_pred] + encoder_states
        )
        """ Decoder (Inference) model """
        encoder_inf_out = Input(shape=(None, latent_dim), name="encoder_inf_out")
        decoder_init_state = Input(shape=(latent_dim,), name="decoder_init")

        decoder_inf_out, decoder_inf_state = decoder_gru(hidden_forecasted_rainfall, initial_state=decoder_init_state)
        attn_inf_out, attn_inf_states = attn_layer([encoder_inf_out, decoder_inf_out])
        decoder_inf_concat = concatenate(axis=-1, name="decoder_inf_concat")([decoder_inf_out, attn_inf_out])
        decoder_inf_pred = decoder_dense_time(decoder_inf_concat)
        self.decoder_model = Model(
            inputs=[input_decoder_streamflow, input_decoder_forecasted_rf, encoder_inf_out, decoder_init_state],
            outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state],
        )

    def fit(self, x, y, x_test, y_test):
        early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="auto")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-4)
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.MODEL_FOLDER, "castle.h5"),
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=False,
        )
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        history = self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            # validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=2,
        )
        # self.model.save(os.path.join(self.MODEL_FOLDER, "castle.h5"))

    #         plt.plot(history.history['loss'])
    #         plt.plot(history.history['val_loss'])
    #         plt.title('model loss')
    #         plt.ylabel('loss')
    #         plt.xlabel('epoch')
    #         plt.legend(['train', 'test'], loc='upper left')
    #         plt.show()

    def predict(self, input_encoder_streamflow, input_encoder_observed_rf, input_decoder_forecasted_rf):
        # Encode the input as state vectors.
        encoder_pred, encoder_state = self.encoder_model.predict([input_encoder_streamflow, input_encoder_observed_rf])
        input_decoder_sf = input_encoder_streamflow[:, -1:, :, :]
        # print(input_decoder_sf.shape, init_prediction[:,-1:].shape)
        input_decoder_sf = np.append(
            input_decoder_sf, encoder_pred[:, -1:].reshape(encoder_pred.shape[0], 1, -1, 1), axis=2
        )
        print(input_decoder_sf.shape)
        input_decoder_sf = input_decoder_sf[:, :, 1:]

        # input_decoder_sf = init_prediction[:, -self.sf_dim :]
        # input_decoder_sf = input_decoder_sf.reshape(input_decoder_sf.shape[0], 1, -1, 1)
        prediction = encoder_pred[:, -1:]
        print(encoder_pred.shape, prediction.shape, input_decoder_sf.shape)
        state_value = encoder_state
        # Sampling loop for a batch of sequences
        attention_weights = []
        for i in range(input_decoder_forecasted_rf.shape[1]):
            # input_decoder_streamflow, input_decoder_forecasted_rf, encoder_inf_states, decoder_init_state
            decoder_pred, attn_states, decoder_state = self.decoder_model.predict(
                [input_decoder_sf, input_decoder_forecasted_rf[:, i : i + 1]] + [encoder_pred, state_value]
            )
            # print("output shape is "+str(output.shape))
            prediction = np.append(prediction, decoder_pred[:, -1:], axis=1)
            decoder_pred = decoder_pred.reshape(decoder_pred.shape[0], 1, -1, 1)
            input_decoder_sf = np.append(input_decoder_sf, decoder_pred, axis=2)
            input_decoder_sf = input_decoder_sf[:, :, 1:]
            # print (prediction.shape, input_decoder_sf.shape)
            attention_weights.append((i + 1, attn_states))
            # Update states
            state_value = decoder_state
        # print(prediction.shape)
        return prediction, attention_weights
