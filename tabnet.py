# train TabNet on our 5 datasets
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
 
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
import os

def save_model(model_name, model):
    if 'models' not in os.listdir('.'):
        os.mkdir('models')

    if 'tabnet' not in os.listdir('models'):
        os.mkdir(os.path.join('models', 'tabnet'))

    model_path = os.path.join(os.path.join('models', 'tabnet', model_name))
    path = model.save_model(model_path)
    return path

def train(X_train, y_train):
    tb_reg = TabNetRegressor(optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax' # "sparsemax"
        )
    
    tb_reg.fit(X_train,y_train, max_epochs=1000 , patience=100, batch_size=28, drop_last=False)
    save_model('future_sales_predict', tb_reg)
    return tb_reg

def predict(tb_cls, X_test, y_test):
    # Test model and generate prediction
    return tb_cls.predict(X_test)

def load_future_sales_dataset():
    items_df = pd.read_csv('datasets/competitive-data-science-predict-future-sales/items.csv')
    item_categories_df=pd.read_csv('datasets/competitive-data-science-predict-future-sales/item_categories.csv')
    shops_df=pd.read_csv('datasets/competitive-data-science-predict-future-sales/shops.csv')
    sales_train_df=pd.read_csv('datasets/competitive-data-science-predict-future-sales/sales_train.csv')
    
    sales_train_df_groupby=sales_train_df.groupby(['shop_id','item_id','date_block_num'])['item_cnt_day'].sum().reset_index()
    
    X = sales_train_df_groupby.drop(columns=['item_cnt_day','date_block_num'])#['date_block_num','shop_id']  # values converts it into a numpy array
    Y = sales_train_df_groupby['item_cnt_day']  # -1 means that calculate the dimension of rows, but have 1 column


    sc = StandardScaler()
    dataset=[]
    dataset = sales_train_df.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
    test_df=pd.read_csv('datasets/competitive-data-science-predict-future-sales/test.csv')
    dataset.reset_index(inplace = True)
    dataset = pd.merge(test_df,dataset,on = ['item_id','shop_id'],how = 'left')
    dataset.fillna(0,inplace = True)
    dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)
    # X we will keep all columns execpt the last one 
    
    X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)
    # the last column is our label
    y_train = dataset.values[:,-1:]

    # for test we keep all the columns execpt the first one
    X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

    return X_train, y_train, X_test

def exp_save():
    X_train, y_train, X_test = load_future_sales_dataset()
    model = train(X_train, y_train)
