import os
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd

def load_eye_movements_dataset(main_data_dir):
    dataset_path = os.path.join(main_data_dir, 'eye-movements')
    Eye_Movements_df=pd.read_csv(os.path.join(dataset_path, 'eye_movements.csv'))
    
    X=Eye_Movements_df.drop(columns=['label', 'lineNo'])
    Y=Eye_Movements_df[['label']]

    X =  X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    return X_train, y_train, X_test, y_test

