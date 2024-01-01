import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,MaxPool1D,MaxPool2D
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

def MLP(X_train, y_train, X_test, y_test):
    
    mlp = MLPClassifier(random_state=1, max_iter=300,activation='logistic')
    start = time.time()
    mlp.fit(X_train,y_train)
    stop = time.time()
    pred = mlp.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    my_model_time=np.round((stop - start)/60, 2)
    return score ,  my_model_time

    

def DNN_model(X_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=X_train.shape[1],activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(8,activation='relu'))
    model.add(tf.keras.layers.Dense(classes,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def Conv1D(X_train):
    model = Sequential()
    model.add(Conv1D(64,(20), padding='same', activation='relu', input_shape=(X_train[1:])))
    model.add(MaxPool1D((1, 1)))
    model.add(Conv2D(8, (1, 1), activation='relu'))
    model.add(MaxPool2D((1, 1)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def Conv2D(X_train):
    model = Sequential()
    model.add(Conv2D(64,(1,1), padding='same', activation='relu', input_shape=(X_train[1:])))
    model.add(MaxPool2D((1, 1)))
    model.add(Conv2D(8, (1, 1), activation='relu'))
    model.add(MaxPool2D((1, 1)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def Conv3D(X_train):
    model = Sequential()
    model.add(Conv2D(64,(1,1), padding='same', activation='relu', iinput_shape=(X_train[1:])))
    model.add(MaxPool2D((1, 1)))
    model.add(Conv2D(8, (1, 1), activation='relu'))
    model.add(MaxPool2D((1, 1)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def DNN_training(X_train, y_train, X_test, y_test):
    model = DNN_model(X_train)

    print('Start to train')
    start = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=( X_test, X_test),shuffle=True)
    stop = time.time()
    my_model_time=np.round((stop - start)/60, 2)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)
    pred = model.predict(X_test)
    pred = np.where(pred>=0.5, 1, 0)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    print("F1-Score :",score)
    return score ,  my_model_time

def Conv_training(X_train, y_train, X_test, y_test,conv_model):
    X_train_conv= np.asarray(X_train).reshape(X_train.shape[0],X_train.shape[1],1)
    X_test_conv = np.asarray(X_test).reshape(X_test.shape[0],X_test.shape[1],1)
    model = conv_model(X_train_conv)
    print('Start to train Conv model')
    start = time.time()
    history = model.fit(X_train_conv, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_conv,y_test),shuffle=True)
    stop = time.time()
    my_model_time=np.round((stop - start)/60, 2)
    my_model_score = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
    vs.train_loss_plot(history)
    pred = model.predict(X_test_conv)
    pred = np.where(pred>=0.5, 1, 0)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    print("F1-Score :",score)
    return score ,  my_model_time

def main_processing():
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