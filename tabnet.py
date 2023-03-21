# train TabNet on our 5 datasets
from pytorch_tabnet.tab_model import TabNetRegressor
 
import os
import torch
from sklearn.preprocessing import RobustScaler,StandardScaler
import os
from sklearn.model_selection import KFold
from LoadFutureSalesDataset import load_future_sales_dataset


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
        seed=42,
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu',
        n_d=8, n_a=8, n_steps=3, gamma=1.3,
        lambda_sparse=0, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type='entmax',
        scheduler_params=dict(mode="min",
                              patience=5,
                              min_lr=1e-5,
                              factor=0.9,),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=10,
    )
    
    tb_reg.fit(X_train,Y_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=10, batch_size=1024, virtual_batch_size=128,
        drop_last=False
    )
    
    return tb_reg

def predict(tb_cls, X_test, y_test):
    # Test model and generate prediction
    return tb_cls.predict(X_test)

def run_tab_net_for_future_sales():
    X, Y, X_test, Y_test = load_future_sales_dataset()

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    CV_score_array = []
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        tb_reg = train(X_train, Y_train, X_val, Y_val)
        CV_score_array.append(tb_reg.best_cost)

        save_model(os.path.join('future_sales', 'model_' + str(i)), tb_reg)

    print(CV_score_array)


run_tab_net_for_future_sales()
