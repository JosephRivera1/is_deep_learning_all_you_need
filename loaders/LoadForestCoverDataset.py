import os
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd

def load_forest_cover_type_dataset(main_data_dir):
    dataset_path = os.path.join(main_data_dir, 'forest-cover-type')
    Forest_Cover_df=pd.read_csv(os.path.join(dataset_path, 'covType.csv'))
    Forest_Cover_df['Cover_Type'] = Forest_Cover_df['Cover_Type'] - 1
    
    X=Forest_Cover_df.drop(columns=['Cover_Type'])
    Y=Forest_Cover_df[['Cover_Type']]

    X =  X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    return X_train, y_train, X_test, y_test

