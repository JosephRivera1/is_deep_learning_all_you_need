import os
from sklearn.model_selection import train_test_split 
import pandas as pd

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
