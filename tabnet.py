# train TabNet on our 5 datasets
from pytorch_tabnet.tab_model import TabNetRegressor
 
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
import os
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold


def save_model(model_name, model):
    if 'models' not in os.listdir('.'):
        os.mkdir('models')

    if 'tabnet' not in os.listdir('models'):
        os.mkdir(os.path.join('models', 'tabnet'))

    model_path = os.path.join(os.path.join('models', 'tabnet', model_name))
    path = model.save_model(model_path)
    return path

def train(X_train, Y_train, X_val, Y_val):
    tb_reg = TabNetRegressor(
        verbose=0,
        seed=42,
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    tb_reg.fit(X_train,Y_train,
          eval_set=[(X_train, Y_train), (X_val, Y_val)],
          eval_name=['train', 'valid'],
          eval_metric=['rmse'],
          max_epochs=10,
          batch_size=32, drop_last=False)
    
    return tb_reg

def predict(tb_cls, X_test, y_test):
    # Test model and generate prediction
    return tb_cls.predict(X_test)

def load_future_sales_dataset():
    data_path = os.path.join('kaggle', 'input', 'competitive-data-science-predict-future-sales')
    sales_train_df=pd.read_csv(os.path.join(data_path, 'sales_train.csv'))
    
    sales_train_df_groupby=sales_train_df.groupby(['shop_id','item_id','date_block_num'])['item_cnt_day'].sum().reset_index()    
    sales_train_df_groupby_sorted = sales_train_df_groupby.sort_values(by=['date_block_num'])

    X = sales_train_df_groupby_sorted.drop(columns=['item_cnt_day','date_block_num'])#['date_block_num','shop_id']  # values converts it into a numpy array
    Y = sales_train_df_groupby_sorted['item_cnt_day']  # -1 means that calculate the dimension of rows, but have 1 column

    X =  X.to_numpy()
    Y = Y.to_numpy()
    
    Y = Y.flatten().reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    return X_train, y_train, X_test, y_test

def run_tab_net_for_future_sales():
    X, Y, X_test, Y_test = load_future_sales_dataset()

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    CV_score_array = []
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        tb_reg = train(X_train, Y_train, X_val, Y_val)
        CV_score_array.append(tb_reg.best_cost)

        save_model('future_sales_predict_' + str(i), tb_reg)

    print(CV_score_array)


run_tab_net_for_future_sales()
