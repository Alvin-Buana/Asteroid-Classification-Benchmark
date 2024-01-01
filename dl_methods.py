import tensorflow as tf

import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import roc_auc_score

import time

import preprocessing_data as dt
import data_visualization as vs


batch_size = 50
classes = 1
epochs = 100
lr = 0.0001

def MLP():
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    mlp = MLPClassifier(random_state=1, max_iter=300,activation='logistic')
    mlp.fit(X_train,y_train)
    pred = mlp.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)

def DNN_model():
    model = tf.keras.models.Sequential(name="model_DNN")
    model.add(tf.keras.layers.Dense(128, input_dim=20,activation='relu'))
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

def DNN_training():
    model = DNN_model()
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    
    data = X_train
    data['Hazardous'] = y_train
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train.drop('Hazardous',axis=1), data_train['Hazardous']
    X_val,y_val = data_val.drop('Hazardous',axis=1), data_val['Hazardous']
    print('Start to train')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)

def Conv1D_training():
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],data.shape[1],1))
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
    vs.train_loss_plot(history)

def Conv2D_training():
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],data.shape[1],1,1))
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train[:,:20,:,:], data_train[:,20]
    print(y_train.shape)
    X_val,y_val = data_val[:,:20,:,:], data_val[:,20]
    print('Start to train')
    n_features = 20
    model = Conv2D(n_features)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)

def RNN_training():
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],1,data.shape[1]))
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train[:,:,:20], data_train[:,:,20]
    X_val,y_val = data_val[:,:,:20], data_val[:,:,20]
    print('Start to train')
    n_timesteps = 20
    model = RNN(n_timesteps)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)

def LSTM_training():
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    data = X_train
    data['Hazardous'] = y_train
    data = data.values
    print(data.shape)
    data = np.reshape(data,(data.shape[0],1,data.shape[1]))
    print(data.shape)
    data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    X_train, y_train = data_train[:,:,:20], data_train[:,:,20]
    X_val,y_val = data_val[:,:,:20], data_val[:,:,20]
    print('Start to train')
    n_timesteps = 20
    model = LSTM(n_timesteps)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val),shuffle=True)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)

def deep_learning_method():
    print('Start to train with ML method')
    print('Start to read data')
    X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    X_train_dl = np.asarray(X_train).reshape((-1,1))
    X_test_dl = np.asarray(X_test).reshape((-1,1))
    # data = X_train
    # data['Hazardous'] = y_train
    # data_train, data_val = train_test_split(data, test_size=0.15, random_state=42)
    # X_train, y_train = data_train.drop('Hazardous',axis=1), data_train['Hazardous']
    # X_val,y_val = data_val.drop('Hazardous',axis=1), data_val['Hazardous']
    f1_score_mlp,time_mlp = MLP(X_train, y_train, X_test, y_test)
    f1_score_dnn,time_dnn = DNN_training(X_train_dl, y_train, X_test_dl, y_test)
    f1_score_conv1d,time_conv1d = Conv_training(X_train_dl, y_train, X_test_dl, y_test,Conv1D)
    f1_score_conv2d,time_conv2d = Conv_training(X_train_dl, y_train, X_test_dl, y_test,Conv2D)
    f1_score_conv3d,time_conv3d = Conv_training(X_train_dl, y_train, X_test_dl, y_test,Conv3D)

    valid_scores=pd.DataFrame(
    {'Classifer':['Logistic Regression','KNN','SVC','Random Forest','LGBM','CatBoost','NaiveBayes'], 
     'Validation F1_score': [f1_score_mlp,f1_score_dnn,f1_score_conv1d,f1_score_conv2d,f1_score_conv3d],  
     'Training time': [time_mlp,time_dnn,time_conv1d,time_conv2d ,time_conv3d],
    })
    filename = "Dl Methods"
    valid_scores.to_csv(filename+'.csv', index=False)
