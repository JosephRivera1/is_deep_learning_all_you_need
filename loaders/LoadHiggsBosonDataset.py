import os
from sklearn.model_selection import train_test_split 
import pandas as pd

def load_higgs_boson_dataset(main_data_dir):
    dataset_path = os.path.join(main_data_dir, 'higgs-boson')
    Higgs_Boson_train_df=pd.read_csv(os.path.join(dataset_path, 'training.csv'))
    
    X=Higgs_Boson_train_df.drop(columns=['EventId', 'Label'])
    Higgs_Boson_train_df['Label'].replace(['s','b'],[0,1],inplace=True)
    Y=Higgs_Boson_train_df[['Label']]

    X =  X.to_numpy()
    Y = Y.to_numpy()

    Y = Y.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    return X_train, y_train, X_test, y_test


