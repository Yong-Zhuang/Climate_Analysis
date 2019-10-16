import os
import pandas as pd
import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import KFold
from tensorflow import set_random_seed, get_seed
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import sarimax

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Conv1D,Flatten, MaxPooling1D, InputLayer
from keras import initializers, regularizers
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'
LOOK_BACK = 15
LEAD = [5,7,10]
TRAIN_TEST = 3825
NUM_EPOCHS = 500
BATCH_SIZE = 50
YEAR_DAYS = 153
PATH_RESULT = 'results/result'
PATH_LOG = 'results/log'

np.random.seed(3) # NumPy
random.seed(3) # Python
set_random_seed(3)

# Helper functions
def load_data(river='g'): #g:Ganges; b: Brahmaputra; m: Meghna;
    #1934-4-01  2018-07-09
    if river =='g':
        river_name = 'Ganges'
    elif river =='b':
        river_name = 'Brahmaputra'
    else:
        river_name = 'Meghna'
    Qx = pd.read_csv('../../data/streamflw_precipitation/X_'+river_name+'.csv', index_col=1,header=0,parse_dates=True)
    X = Qx.iloc[:, -LOOK_BACK:]
    Qy = pd.read_csv('../../data/streamflw_precipitation/Y_'+river_name+'.csv', index_col=1,header=0,parse_dates=True)
    #print (Qy.head())
    idy = []
    for i in LEAD:
        idy.append('Q_'+str(i))
    y = Qy.loc[:,idy]
    return X, y

def initialization():
    if os.path.isdir(PATH_RESULT) is False:
        os.mkdir(PATH_RESULT)
    if os.path.isdir(PATH_LOG) is False:
        os.mkdir(PATH_LOG)

def get_metrics(y, pred):
    m_mae = metrics.mean_absolute_error(y, pred)
    m_rmse = metrics.mean_squared_error(y, pred)** 0.5
    m_r2 = metrics.r2_score(y, pred) 
    return m_mae,m_rmse,m_r2    

# Models
def build_ann():
    ann = Sequential()
    ann.add(InputLayer((LOOK_BACK, 1)))
    ann.add(Flatten())
    ann.add(Dense(100, activation='relu'))
    ann.add(Dense(50, activation='relu'))
    ann.add(Dense(1))
    ann.compile(optimizer='adam', loss='mse')
    ann.summary()
    return ann

def build_cnn():
    cnn = Sequential()
    cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(LOOK_BACK, 1)))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Flatten())
    cnn.add(Dense(100, activation='relu'))
    cnn.add(Dense(1))
    cnn.compile(optimizer='adam', loss='mse')
    cnn.summary()
    return cnn
def build_rnn():
    rnn = Sequential()
    rnn.add(SimpleRNN(50, activation='relu', input_shape=(LOOK_BACK, 1)))
    rnn.add(Dense(100, activation='relu'))
    rnn.add(Dense(1))
    rnn.compile(optimizer='adam', loss='mse')
    rnn.summary()
    return rnn
def build_lstm():
    lstm = Sequential()
    lstm.add(LSTM(50, activation='relu', input_shape=(LOOK_BACK, 1)))
    lstm.add(Dense(100, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.summary()
    return lstm
    
models = [
        ['ARIMA', ARIMA],
        ['SARIMAX', sarimax.SARIMAX],
        ['ANN', build_ann],
        ['CNN', build_cnn],
        ['RNN', build_rnn],
        ['LSTM', build_lstm]]


if __name__== "__main__":     
    data_X, data_y = load_data('g')
    initialization()
    X = data_X.values[:,:,np.newaxis]
    performance_y_test = {}
    performance_y_test[5]={}
    performance_y_test[5]["MORN"]={}
    performance_y_test[5]["MORN"]["MAE"] = 2435
    performance_y_test[5]["MORN"]["RMSE"] = 3567
    performance_y_test[5]["MORN"]["R2"] = 0.94
    performance_y_test[7]={}
    performance_y_test[7]["MORN"]={}
    performance_y_test[7]["MORN"]["MAE"] = 3028
    performance_y_test[7]["MORN"]["RMSE"] = 4389
    performance_y_test[7]["MORN"]["R2"] = 0.91
    performance_y_test[10]={}
    performance_y_test[10]["MORN"]={}
    performance_y_test[10]["MORN"]["MAE"] = 3580
    performance_y_test[10]["MORN"]["RMSE"] = 5367
    performance_y_test[10]["MORN"]["R2"] = 0.871
    n_splits = 3
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    for lead in LEAD:
        print ("%%%%%%%%%%%%%%%%%%%% start experiments with lead time "+str(lead)+" %%%%%%%%%%%%%%%%%%%%")
        y = data_y.loc[:,'Q_'+str(lead)].values 
        X_train, y_train, X_test, y_test = X[:TRAIN_TEST], y[:TRAIN_TEST], X[TRAIN_TEST:], y[TRAIN_TEST:]    
        for name, model in models:
            mae=0
            rmse=0
            r2=0
            performance_y_test[lead][name]={}
            if name == "ARIMA":            
                prediction = list()
                for t in X_test:
                    #print (t.shape)
                    clf = model(t, order=(1,1,0))
                    clf_fit = clf.fit()
                    yhat = clf_fit.forecast(steps=lead)[0][-1]
                    prediction.append(yhat)
                m_mae,m_rmse,m_r2 = get_metrics(np.array(y_test),prediction)
                performance_y_test[lead][name]["MAE"] = m_mae
                performance_y_test[lead][name]["RMSE"] = m_rmse
                performance_y_test[lead][name]["R2"] = m_r2
                print (m_mae,m_rmse,m_r2)
            elif name == 'SARIMAX':
                prediction = list()
                for index, t in enumerate(X_test):
                    total_index = TRAIN_TEST+index
                    history = X[(total_index-3*YEAR_DAYS):total_index,-1]                 
                    print (history.shape)
                    clf = model(history, order=(1,1,0), seasonal_order=(1, 1, 0, YEAR_DAYS),enforce_stationarity=False,enforce_invertibility = False, simple_differencing = True)
                    clf_fit = clf.fit()
                    yhat = clf_fit.forecast(steps=lead)[-1]
                    prediction.append(yhat)
                m_mae,m_rmse,m_r2 = get_metrics(np.array(y_test),prediction)
                performance_y_test[lead][name]["MAE"] = m_mae
                performance_y_test[lead][name]["RMSE"] = m_rmse
                performance_y_test[lead][name]["R2"] = m_r2
                print (m_mae,m_rmse,m_r2)
            else:
                for train, validation in cv.split(X_train, y_train):
                    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4)    
                    callbacks = [early_stopping,reduce_lr]
                    clf = model()
                    history = clf.fit(X_train[train],y_train[train],
                                epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(X_train[validation],y_train[validation]),               
                                callbacks=callbacks,
                                verbose=1)
                    prediction = clf.predict(X_test,verbose=0)
                    m_mae,m_rmse,m_r2 = get_metrics(np.array(y_test),prediction)
                    mae +=m_mae
                    rmse +=m_rmse
                    r2 +=m_r2
                performance_y_test[lead][name]["MAE"] = mae/n_splits
                performance_y_test[lead][name]["RMSE"] = rmse/n_splits
                performance_y_test[lead][name]["R2"] = r2/n_splits

    df_result = pd.DataFrame.from_dict({(i,j): performance_y_test[i][j] 
                               for i in performance_y_test.keys() 
                               for j in performance_y_test[i].keys()},
                           orient='index')
    df_result.index = df_result.index.set_names(['Lead','Model'])
    df_result.to_csv(PATH_RESULT+'/results1.csv')