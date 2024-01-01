# Models
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import roc_auc_score

import time
import numpy as np
import pandas as pd 

import preprocessing_data as dt

def logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=500)
    start = time.time()
    clf.fit(X_train,y_train)
    stop = time.time()
    time_total=np.round((stop - start)/60, 2)
    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',time_total)
    return score, roc_score, time_total

def knn(X_train, y_train, X_test, y_test):
    if X_train.shape[1] >20:
        pca = PCA(n_components = 5)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    best_score ={}
    for k in range(2,10):
        knnclassifier = KNeighborsClassifier(n_neighbors=k)
        knnclassifier.fit(X_train,y_train)
        pred = knnclassifier.predict(X_test)
        conmat = confusion_matrix(y_test,pred)
        score = f1_score(y_test,pred)
        roc_score = roc_auc_score(y_test,pred)
        best_score[k] = score
    print("The best k value is :",max(best_score),"with a score of",max(best_score.values()))

    clf = KNeighborsClassifier(n_neighbors=max(best_score))
    start = time.time()
    clf.fit(X_train,y_train)
    stop = time.time()
    time_total=np.round((stop - start)/60, 2)

    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',time_total)
    return score, roc_score, time_total




def svc(X_train, y_train, X_test, y_test):
    classifier = SVC(random_state=0, probability=True)
    SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']}
    clf = GridSearchCV(estimator=classifier, param_grid=SVC_grid, n_jobs=-1, cv=None)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    time_total=np.round((stop - start)/60, 2)

    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',time_total)
    return score, roc_score, time_total

def random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(random_state=0)
    RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [4, 6, 8, 10, 12]}
    clf = GridSearchCV(estimator=classifier, param_grid=RF_grid, n_jobs=-1, cv=None)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    time_total=np.round((stop - start)/60, 2)

    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',time_total)
    return score, roc_score, time_total

def lgbm(X_train, y_train, X_test, y_test):
    classifier = LGBMClassifier(random_state=0)
    boosted_grid = {'n_estimators': [50, 100, 150, 200],
            'max_depth': [4, 8, 12],
            'learning_rate': [0.05, 0.1, 0.15]}
    clf = GridSearchCV(estimator=classifier, param_grid=boosted_grid, n_jobs=-1, cv=None)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    LGBM_time=np.round((stop - start)/60, 2)
    
    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',LGBM_time)
    return score, roc_score, LGBM_time

def catboost(X_train, y_train, X_test, y_test):
    classifier = CatBoostClassifier(random_state=0, verbose=False)
    boosted_grid = {'n_estimators': [50, 100, 150, 200],
            'max_depth': [4, 8, 12],
            'learning_rate': [0.05, 0.1, 0.15]}
    clf = GridSearchCV(estimator=classifier, param_grid=boosted_grid, n_jobs=-1, cv=None)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    CB_time=np.round((stop - start)/60, 2)
    # CB_score = clf.score(X_valid, y_valid)

    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)
    print('Training time (mins):',CB_time)
    return score, roc_score, CB_time

def naive_bayes(X_train, y_train, X_test, y_test):
    classifier = GaussianNB()
    NB_grid={'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}
    clf = GridSearchCV(estimator=classifier, param_grid=NB_grid, n_jobs=-1, cv=None)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    NB_time=np.round((stop - start)/60, 2)
    # NB_score = clf.score(X_valid, y_valid)

    pred = clf.predict(X_test)
    conmat = confusion_matrix(y_test,pred)
    score = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred)
    print("F1-Score :",score)
    print("ROC Score :",roc_score)    
    print('Training time (mins):',NB_time)
    return score, roc_score, NB_time


def machine_learning_method(X_train, y_train, X_test, y_test,filename):
    # print('Start to train with ML method')
    # print('Start to read data')
    # X_train, y_train, X_test, y_test = dt.read_data_NeoWS()
    # print('Start to train')

    f1_score_lg, roc_score_lg, time_lg =        logistic_regression(X_train, y_train, X_test, y_test)
    f1_score_knn, roc_score_knn, time_knn =     knn(X_train, y_train, X_test, y_test)
    f1_score_svc, roc_score_svc, time_svc =     svc(X_train, y_train, X_test, y_test)
    f1_score_rf, roc_score_rf, time_rf =        random_forest(X_train, y_train, X_test, y_test)
    f1_score_lgbm, roc_score_lgbm, time_lgbm =  lgbm(X_train, y_train, X_test, y_test)
    f1_score_cb, roc_score_cb, time_cb =        catboost(X_train, y_train, X_test, y_test)
    f1_score_nb, roc_score_nb, time_nb =        naive_bayes(X_train, y_train, X_test, y_test)
    

    valid_scores=pd.DataFrame(
    {'Classifer':['Logistic Regression','KNN','SVC','Random Forest','LGBM','CatBoost','NaiveBayes'], 
     'Validation F1_score': [f1_score_lg,f1_score_knn,f1_score_svc,f1_score_rf,f1_score_lgbm,f1_score_cb,f1_score_nb],
     'Validation ROC Score': [roc_score_lg,roc_score_knn,roc_score_svc,roc_score_rf,roc_score_lgbm,roc_score_cb,roc_score_nb],  
     'Training time': [time_lg,time_knn,time_svc,time_rf,time_lgbm,time_cb,time_nb],
    })
    valid_scores.to_csv(filename+'.csv', index=False)
    

def main_processing(data,imbalance):
    print('Start to train with ML method')
    print('Start to read data')
    if data == "NeoWs":
        X_train, y_train, X_test, y_test = dt.read_data_NeoWS(imbalance)
    else:
        X_train, y_train, X_test, y_test =dt.read_data_Asteroid(imbalance)
    filename = 'ML_'+data+'_'+imbalance+'_methods'
    machine_learning_method(X_train, y_train, X_test, y_test,filename)

    print("Done")
    

