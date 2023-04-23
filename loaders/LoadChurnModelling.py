import os
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_churn_modelling_dataset(main_data_dir):
    dataset_path = os.path.join(main_data_dir, 'churn-modelling')
    Churn_Modelling_df=pd.read_csv(os.path.join(dataset_path, 'Churn_Modelling.csv'))

    label_encoder = LabelEncoder()
    Churn_Modelling_df['Surname'] = label_encoder.fit_transform(Churn_Modelling_df['Surname'])

    label_encoder = LabelEncoder()
    Churn_Modelling_df['Geography'] = label_encoder.fit_transform(Churn_Modelling_df['Geography'])

    label_encoder = LabelEncoder()
    Churn_Modelling_df['Gender'] = label_encoder.fit_transform(Churn_Modelling_df['Gender'])

    X=Churn_Modelling_df.drop(columns=['RowNumber', 'CustomerId', 'Exited'])
    Y=Churn_Modelling_df[['Exited']]

    X =  X.to_numpy()
    Y = Y.to_numpy().flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    return X_train, y_train, X_test, y_test

