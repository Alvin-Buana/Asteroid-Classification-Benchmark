import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

datasets= ["Asteroids","NeoWS"]
dataset_path = 'dataset/'

def normalization(data):
    scaler = MinMaxScaler()
    numerical_cols = data.select_dtypes(include='number').columns
    for num in numerical_cols:
        data[num] =scaler.fit_transform(np.array(data[num]).reshape(-1,1))
    return data

def categorical(data):
    categorical_cols = data.select_dtypes(include='object').columns
    encoder = LabelEncoder()
    for cat in categorical_cols:
        data[cat] = encoder.transform(data[cat])
    return data


def data_cleaning(data):
    data.dropna(axis=0,inplace=True)
    data.drop_duplicates(inplace=True)
    columns = [ 'Date','Neo Reference ID','Name','Equinox','Close Approach Date','Epoch Date Close Approach', 
                'Orbit Determination Date','Orbiting Body','Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 
                'Est Dia in Miles(min)','Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
                'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)','Miss Dist.(kilometers)', 'Miss Dist.(miles)']
    data.drop(columns, axis = 1,inplace=True)
    return data

def imbalance_bootstrapping(data, n_samples):
    data_minority = data[data['Hazardous']==True]
    data_majority = data[data['Hazardous']==False]
    data_minority = resample(data_minority,random_state=42,n_samples=n_samples)
    data_majority = resample(data_majority,random_state=42,n_samples=n_samples)
    data = pd.concat([data_minority,data_majority],axis=0)
    return data

def imbalance_smote(data):
    resampler = SMOTE(random_state=42)
    X_data = data.drop(['Hazardous'],axis=1)
    y_data = data['Hazardous'].values
    X_smote_train,y_smote_train = resampler.fit_resample(X_data,y_data)
    data = X_smote_train.copy()
    data['Hazardous'] = y_smote_train
    return data

def read_raw_data_NeoWS():
    path = dataset_path + datasets[1]
    data1900_1950 = pd.read_csv(path+'/data_1900_1950_updated_December.csv',index_col=0).reset_index(drop=True)
    data1950_1953 = pd.read_csv(path+'/data_1950_2000.csv',index_col=0).reset_index(drop=True)
    data1953_2000 = pd.read_csv(path+'/data_1953_2000.csv',index_col=0).reset_index(drop=True)
    data2002_2022 = pd.read_csv(path+'/data_2002_2022.csv',index_col=0).reset_index(drop=True)
    data2022_present = pd.read_csv(path+'/data_2022_present.csv',index_col=0).reset_index(drop=True)
    data = pd.concat([data1900_1950,data1950_1953,data1953_2000,data2002_2022,data2022_present],axis=0)
    return data 

def read_data_NeoWS():
    data = read_raw_data_NeoWS()
    data = data_cleaning(data)
    print(data.head())
    #exit()
    data = imbalance_bootstrapping(data,10000)
    data = normalization(data)
    data = categorical(data)
    data_train,data_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = data_train.drop(['Hazardous'],axis=1)
    y_train = data_train['Hazardous']
    X_test = data_test.drop(['Hazardous'],axis=1)
    y_test = data_test['Hazardous']
    return X_train, y_train, X_test, y_test

