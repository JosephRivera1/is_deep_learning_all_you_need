from tabnet import TabNetRegCustom, TabNetCfCustom, TabNetMultiCfCustom
from sklearn.model_selection import KFold, train_test_split
from loaders.LoadFutureSalesDataset import load_future_sales_dataset
from loaders.LoadHiggsBosonDataset import load_higgs_boson_dataset
from loaders.LoadForestCoverDataset import load_forest_cover_type_dataset
from loaders.LoadEyeMovementsDataset import load_eye_movements_dataset
from loaders.LoadChurnModelling import load_churn_modelling_dataset
import os
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

import torch
import torch.nn as nn

import warnings


def run_tab_net_for_future_sales():
    X, Y, X_test, Y_test = load_future_sales_dataset(os.path.join(os.curdir, 'datasets'))
    tbReg = TabNetRegCustom('model_future_sales')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('--------------------- TRAINING TABNET FOR FUTURE SALES DATASET -----------------------------------')
    
    val_loss = tbReg.train(X_train, y_train, X_val, y_val)
    
    print('RMSE val loss:', val_loss)

    preds = tbReg.predict(X_test)
    print('RMSE test: %f' % (np.sqrt(mean_squared_error(Y_test, preds))))
    

def run_tab_net_for_higgs_boson():
    X, Y, X_test, Y_test = load_higgs_boson_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetCfCustom('model_higgs_boson')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('--------------------- TRAINING TABNET FOR HIGGS BOSON DATASET -----------------------------------')

    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)
    
    preds = tbCf.predict(X_test)
    logLoss = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    acc = accuracy_score(Y_test, preds)
    print("Test cross-entropy: %f" % (logLoss))
    print("Test-accuracy: %f" % (acc))


def run_tab_net_for_forest_cover_type():
    X, Y, X_test, Y_test = load_forest_cover_type_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetMultiCfCustom('model_forest_cover_type')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING TABNET FOR FOREST COVER TYPE DATASET -----------------------------------')

    val_loss = tbCf.train(X_train, y_train, X_val, y_val)

    print('Val cross-entropy loss:', val_loss)
    
    preds = tbCf.predict(X_test).transpose().flatten().astype(np.int64)
    Y_test = Y_test.flatten()
    logLoss = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)[0]), torch.tensor(Y_test))
    acc = accuracy_score(Y_test, preds)
    print("Test cross-entropy: %f" % (logLoss))
    print("Test-accuracy: %f" % (acc))

def run_tab_net_for_eye_movements():
    X, Y, X_test, Y_test = load_eye_movements_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetMultiCfCustom('model_eye_movements', 1000)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING TABNET FOR EYE MOVEMENTS DATASET -----------------------------------')

    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)
    
    preds = tbCf.predict(X_test).transpose().flatten().astype(np.int64)
    Y_test = Y_test.flatten()
    logLoss = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)[0]), torch.tensor(Y_test))
    acc = accuracy_score(Y_test, preds)
    print("Test cross-entropy: %f" % (logLoss))
    print("Test-accuracy: %f" % (acc))

def run_tab_net_for_churn_modelling():
    X, Y, X_test, Y_test = load_churn_modelling_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetCfCustom('model_churn_modelling', max_epochs=1000)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING TABNET FOR CHURN MODELLING DATASET -----------------------------------')

    val_loss = tbCf.train(X_train, y_train, X_val, y_val)

    print('Val cross-entropy loss:', val_loss)
    
    preds = tbCf.predict(X_test)
    logLoss = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    acc = accuracy_score(Y_test, preds)
    print("Test cross-entropy: %f" % (logLoss))
    print("Test-accuracy: %f" % (acc))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    run_tab_net_for_future_sales()
    print()

    run_tab_net_for_higgs_boson()
    print()

    run_tab_net_for_forest_cover_type()
    print()

    run_tab_net_for_eye_movements()
    print()

    run_tab_net_for_churn_modelling()