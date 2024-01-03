import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,f1_score ,accuracy_score
from sklearn.metrics import roc_auc_score

import time

import preprocessing_data as dt
import data_visualization as vs


batch_size = 50
classes = 1
epochs = 100
lr = 0.0001

def MLP(X_train, y_train, X_test, y_test):
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    mlp = MLPClassifier(random_state=1, max_iter=300,activation='logistic')
    mlp.fit(X_train,y_train)
    pred = mlp.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    print("F1-Score :",score)
    print("Accuracy Score :",accuracy)
    return score,accuracy

def DNN_model(dim):
    model = tf.keras.models.Sequential(name="model_DNN")
    model.add(tf.keras.layers.Dense(128, input_dim=dim,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(8,activation='relu'))
    model.add(tf.keras.layers.Dense(classes,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def Conv1D(n_timesteps,n_features):
    model = Sequential()
    model = tf.keras.models.Sequential(name="model_Conv1D")
    model.add(tf.keras.layers.Input(shape=(n_timesteps,n_features)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(tf.keras.layers.Dense(n_features, name="Dense_2"))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def Conv2D(n_features):
    model = tf.keras.models.Sequential(name="model_Conv2D")
    model.add(tf.keras.layers.Input(shape=(n_features,1,1)))
    model.add(tf.keras.layers.Conv2D(64,kernel_size = (2,1), padding='same', activation='relu',name="Conv2D_1"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(32,kernel_size = (2,1), padding='same', activation='relu',name="Conv2D_2"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(16,kernel_size = (2,1), padding='same', activation='relu',name="Conv2D_3"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1),name="MaxPooling2D"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def RNN(n_timesteps):
    model = Sequential()
    model.add(tf.keras.layers.SimpleRNN(64, input_shape=(None, n_timesteps)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(tf.keras.layers.Dense(1, name="Dense_2"))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def LSTM(n_timesteps):
    model = Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=(None, n_timesteps)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(tf.keras.layers.Dense(1, name="Dense_2"))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def evaluation_metrics(model,X_test,y_test):
    pred = model.predict(X_test)
    pred = np.where(pred>=0.5,1,0)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    print("F1-Score :",score)
    print("Accuracy Score :",accuracy)
    return score,accuracy
    

def DNN_training(X_train, y_train, X_test, y_test):
    model = DNN_model(X_train.shape[1])
    print('Start to train with ML method')
    print('Start to read data')
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    
    data = X_train
    data['Hazardous'] = y_train
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train.drop('Hazardous',axis=1), data_train['Hazardous']
    X_val,y_val = data_val.drop('Hazardous',axis=1), data_val['Hazardous']
    print('Start to train')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history,"DNN")
    return evaluation_metrics(model,X_test,y_test)
    


def Conv1D_training(X_train, y_train, X_test, y_test):
    print('Start to train with ML method')
    print('Start to read data')
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],data.shape[1],1))
    X_test = X_test.reshape(X_test[0],X_test[1],1)
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train[:,:20,:], data_train[:,20,:]
    X_val,y_val = data_val[:,:20,:], data_val[:,20,:]
    print('Start to train')
    n_timesteps = 20
    n_features = 1
    model = Conv1D(n_timesteps,n_features)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history,"Conv1D")

    return evaluation_metrics(model,X_test,y_test)

def Conv2D_training(X_train, y_train, X_test, y_test):
    print('Start to train with ML method')
    print('Start to read data')
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],data.shape[1],1,1))
    X_test = np.array(X_test).reshape(X_test.shape[0],X_test.shape[1],1,1)
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    size = X_train.shape[1]-1
    X_train, y_train = data_train[:,:size,:,:], data_train[:,size]
    print(X_train.shape)
    print(y_train.shape)
    X_val,y_val = data_val[:,:size,:,:], data_val[:,size]
    print(X_val.shape)
    print(y_val.shape)
    print('Start to train')
    n_features = X_train.shape[1]
    model = Conv2D(n_features)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history,"Conv2D")
    return evaluation_metrics(model,X_test,y_test)

def RNN_training(X_train, y_train, X_test, y_test):
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],1,data.shape[1]))
    X_test = np.array(X_test).reshape(X_test.shape[0],1,X_test.shape[1])
    
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    size =X_train.shape[1]-1
    
    X_train, y_train = data_train[:,:,:size], data_train[:,:,size]
    X_val,y_val = data_val[:,:,:size], data_val[:,:,size]
    print(X_train.shape)
    print(X_val.shape)
    print('Start to train')
    n_timesteps = size
    model = RNN(n_timesteps)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history,"RNN")
    return evaluation_metrics(model,X_test,y_test)

def LSTM_training(X_train, y_train, X_test, y_test):
    print('Start to train with ML method')
    print('Start to read data')
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],1,data.shape[1]))
    X_test = np.array(X_test).reshape(X_test.shape[0],1,X_test.shape[1])
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    size =X_train.shape[1]-1
    
    X_train, y_train = data_train[:,:,:size], data_train[:,:,size]
    X_val,y_val = data_val[:,:,:size], data_val[:,:,size]
    print(X_train.shape)
    print(X_val.shape)
    print('Start to train')
    n_timesteps = size
    model = LSTM(n_timesteps)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history,"LSTM")
    return evaluation_metrics(model,X_test,y_test)

def deep_learning_method(data,imbalance):
    print('Start to train with ML method')
    print('Start to read data')
    if data == "NeoWs":
        X_train, y_train, X_test, y_test = dt.read_data_NeoWS(imbalance)
    else:
        X_train, y_train, X_test, y_test =dt.read_data_Asteroid(imbalance)
    X_train_dl = np.asarray(X_train).reshape((-1,1))
    X_test_dl = np.asarray(X_test).reshape((-1,1))
    print(X_train)
    f1_score_mlp,accuracy_mlp = MLP(X_train, y_train, X_test, y_test)
    f1_score_dnn,accuracy_dnn = DNN_training(X_train, y_train, X_test, y_test)
    f1_score_conv2d,accuracy_conv2d = Conv2D_training(X_train, y_train, X_test, y_test)
    f1_score_RNN,accuracy_RNN= RNN_training(X_train, y_train, X_test, y_test)
    f1_score_LSTM,accuracy_LSTM = LSTM_training(X_train, y_train, X_test, y_test)

    valid_scores=pd.DataFrame(
    {'Classifer':['MLP','DNN','Conv2D','RNN','LSTM'], 
     'Validation F1_score': [f1_score_mlp,f1_score_dnn,f1_score_conv2d,f1_score_RNN,f1_score_LSTM],  
     'Training accuracy': [accuracy_mlp,accuracy_dnn,accuracy_conv2d,accuracy_RNN ,accuracy_LSTM],
    })
    filename = 'DL_'+data+'_'+imbalance+'_methods'
    valid_scores.to_csv(filename+'.csv', index=False)
