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

import xgboost as xgb
import torch
import torch.nn as nn

import warnings


def run_tab_net_for_future_sales():
    X, Y, X_test, Y_test = load_future_sales_dataset(os.path.join(os.curdir, 'datasets'))
    tbReg = TabNetRegCustom('model_future_sales')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('--------------------- TRAINING ON FUTURE SALES DATASET -----------------------------------')
    print('Training Tabnet:')
    val_loss = tbReg.train(X_train, y_train, X_val, y_val)
    
    print('RMSE val loss:', val_loss)

    print('Training XGBoost')
    eval_set = [(X_val, y_val)]
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
    xg_reg.fit(X_train,y_train, eval_set=eval_set, eval_metric='rmse', verbose=True)

    predsTabnet = tbReg.predict(X_test)
    print('RMSE test Tabnet: %f' % (np.sqrt(mean_squared_error(Y_test, predsTabnet))))
    
    predsXGB = xg_reg.predict(X_test)
    print('RMSE test XGBoost: %f' % (np.sqrt(mean_squared_error(Y_test, predsXGB))))

    averaged_preds = (0.5 * predsXGB) + (0.5 * predsTabnet)
    print('RMSE test averaged preds: %f' % (np.sqrt(mean_squared_error(Y_test, averaged_preds))))

def run_tab_net_for_higgs_boson():
    X, Y, X_test, Y_test = load_higgs_boson_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetCfCustom('model_higgs_boson')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('--------------------- TRAINING ON HIGGS BOSON DATASET -----------------------------------')
    print('Training Tabnet:')
    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)

    print('Training XGBoost')
    eval_set = [(X_val, y_val)]
    xg_cf = xgb.XGBClassifier()
    xg_cf.fit(X_train,y_train, eval_set=eval_set, eval_metric='logloss', verbose=True)
    
    predsTabnet = tbCf.predict(X_test)
    logLossTabnet = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using Tabnet: %f" % (logLossTabnet))
    print("Test-accuracy using Tabnet: %f" % (accuracy_score(Y_test, predsTabnet)))

    predsXGB = xg_cf.predict(X_test)
    logLossXGB = nn.functional.cross_entropy(torch.tensor(xg_cf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using XGBoost: %f" % (logLossXGB))
    print("Test-accuracy using XGBoost: %f" % (accuracy_score(Y_test, predsXGB)))

    averaged_probs = (0.5 * xg_cf.predict_proba(X_test)) + (0.5 * tbCf.predict_proba(X_test))
    averaged_preds = np.argmax(averaged_probs, axis=1)
    logLossAvg = nn.functional.cross_entropy(torch.tensor(averaged_probs), torch.tensor(Y_test))
    print("Test cross-entropy using averaeed models: %f" % (logLossAvg))
    print("Test-accuracy using averaged models: %f" % (accuracy_score(Y_test, averaged_preds)))


def run_tab_net_for_forest_cover_type():
    X, Y, X_test, Y_test = load_forest_cover_type_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetMultiCfCustom('model_forest_cover_type')
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING ON FOREST COVER TYPE DATASET -----------------------------------')
    print('Training Tabnet:')
    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)

    print('Training XGBoost')
    eval_set = [(X_val, y_val)]
    xg_cf = xgb.XGBClassifier()
    xg_cf.fit(X_train,y_train, eval_set=eval_set, eval_metric='mlogloss', verbose=True)
    
    predsTabnet = tbCf.predict(X_test)
    logLossTabnet = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using Tabnet: %f" % (logLossTabnet))
    print("Test-accuracy using Tabnet: %f" % (accuracy_score(Y_test, predsTabnet)))

    predsXGB = xg_cf.predict(X_test)
    logLossXGB = nn.functional.cross_entropy(torch.tensor(xg_cf.predict_proba(X_test)), torch.tensor(Y_test.flatten()))
    print("Test cross-entropy using XGBoost: %f" % (logLossXGB))
    print("Test-accuracy using XGBoost: %f" % (accuracy_score(Y_test, predsXGB)))

    averaged_probs = (0.5 * xg_cf.predict_proba(X_test)) + (0.5 * tbCf.predict_proba(X_test))
    averaged_preds = np.argmax(averaged_probs, axis=1)
    logLossAvg = nn.functional.cross_entropy(torch.tensor(averaged_probs), torch.tensor(Y_test))
    print("Test cross-entropy using averaeed models: %f" % (logLossAvg))
    print("Test-accuracy using averaged models: %f" % (accuracy_score(Y_test, averaged_preds)))

def run_tab_net_for_eye_movements():
    X, Y, X_test, Y_test = load_eye_movements_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetMultiCfCustom('model_eye_movements', 1000)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING ON EYE MOVEMENTS DATASET -----------------------------------')
    print('Training Tabnet:')
    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)

    print('Training XGBoost')
    eval_set = [(X_val, y_val)]
    xg_cf = xgb.XGBClassifier()
    xg_cf.fit(X_train,y_train, eval_set=eval_set, eval_metric='mlogloss', verbose=True)
    
    predsTabnet = tbCf.predict(X_test)
    logLossTabnet = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using Tabnet: %f" % (logLossTabnet))
    print("Test-accuracy using Tabnet: %f" % (accuracy_score(Y_test, predsTabnet)))

    predsXGB = xg_cf.predict(X_test)
    logLossXGB = nn.functional.cross_entropy(torch.tensor(xg_cf.predict_proba(X_test)), torch.tensor(Y_test.flatten()))
    print("Test cross-entropy using XGBoost: %f" % (logLossXGB))
    print("Test-accuracy using XGBoost: %f" % (accuracy_score(Y_test, predsXGB)))

    averaged_probs = (0.5 * xg_cf.predict_proba(X_test)) + (0.5 * tbCf.predict_proba(X_test))
    averaged_preds = np.argmax(averaged_probs, axis=1)
    logLossAvg = nn.functional.cross_entropy(torch.tensor(averaged_probs), torch.tensor(Y_test))
    print("Test cross-entropy using averaeed models: %f" % (logLossAvg))
    print("Test-accuracy using averaged models: %f" % (accuracy_score(Y_test, averaged_preds)))

def run_tab_net_for_churn_modelling():
    X, Y, X_test, Y_test = load_churn_modelling_dataset(os.path.join(os.curdir, 'datasets'))
    tbCf = TabNetCfCustom('model_churn_modelling', max_epochs=1000)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('--------------------- TRAINING ON CHURN MODELLING DATASET -----------------------------------')
    print('Training Tabnet:')
    val_loss = tbCf.train(X_train, y_train, X_val, y_val)
    
    print('Val cross-entropy loss:', val_loss)

    print('Training XGBoost')
    eval_set = [(X_val, y_val)]
    xg_cf = xgb.XGBClassifier()
    xg_cf.fit(X_train,y_train, eval_set=eval_set, eval_metric='logloss', verbose=True)
    
    predsTabnet = tbCf.predict(X_test)
    logLossTabnet = nn.functional.cross_entropy(torch.tensor(tbCf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using Tabnet: %f" % (logLossTabnet))
    print("Test-accuracy using Tabnet: %f" % (accuracy_score(Y_test, predsTabnet)))

    predsXGB = xg_cf.predict(X_test)
    logLossXGB = nn.functional.cross_entropy(torch.tensor(xg_cf.predict_proba(X_test)), torch.tensor(Y_test))
    print("Test cross-entropy using XGBoost: %f" % (logLossXGB))
    print("Test-accuracy using XGBoost: %f" % (accuracy_score(Y_test, predsXGB)))

    averaged_probs = (0.5 * xg_cf.predict_proba(X_test)) + (0.5 * tbCf.predict_proba(X_test))
    averaged_preds = np.argmax(averaged_probs, axis=1)
    logLossAvg = nn.functional.cross_entropy(torch.tensor(averaged_probs), torch.tensor(Y_test))
    print("Test cross-entropy using averaeed models: %f" % (logLossAvg))
    print("Test-accuracy using averaged models: %f" % (accuracy_score(Y_test, averaged_preds)))

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