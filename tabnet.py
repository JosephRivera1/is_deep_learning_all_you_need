# train TabNet on our 5 datasets
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
 
import torch
import os
import numpy as np

from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn

class TabNetRegCustom:
    def __init__(self, model_name, max_epochs=100):
        self.model_name = model_name
        self.max_epochs = max_epochs

    def save_model(self, model):
        if 'models' not in os.listdir('.'):
            os.mkdir('models')

        if 'tabnet' not in os.listdir('models'):
            os.mkdir(os.path.join('models', 'tabnet'))

        model_path = os.path.join(os.path.join('models', 'tabnet', self.model_name))
        path = model.save_model(model_path)
        return path
    
    def load_model(self):
        if 'models' not in os.listdir('.') or 'tabnet' not in os.listdir('models'):
            return None
        
        if self.model_name + '.zip' not in os.listdir(os.path.join('models', 'tabnet')):
            return None

        model_path = os.path.join('models', 'tabnet', self.model_name + '.zip')
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
        tb_reg.load_model(model_path)
        return tb_reg
        
    def predict(self, X_test):
        model = self.load_model()
        if not model:
            return None
        
        return np.array(model.predict(X_test))

    def train(self, X_train, Y_train, X_val, Y_val):
        model = self.load_model()
        if model is not None:
            return np.sqrt(mean_squared_error(Y_val, np.array(model.predict(X_val))))

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
            max_epochs=self.max_epochs,
            patience=int(self.max_epochs/10), batch_size=1024, virtual_batch_size=128,
            drop_last=False
        )

        self.save_model(tb_reg)
        
        return tb_reg.best_cost
    

class TabNetCfCustom:
    def __init__(self, model_name, max_epochs = 100):
        self.model_name = model_name
        self.max_epochs = max_epochs

    def save_model(self, model):
        if 'models' not in os.listdir('.'):
            os.mkdir('models')

        if 'tabnet' not in os.listdir('models'):
            os.mkdir(os.path.join('models', 'tabnet'))

        model_path = os.path.join(os.path.join('models', 'tabnet', self.model_name))
        path = model.save_model(model_path)
        return path
    
    def load_model(self):
        if 'models' not in os.listdir('.') or 'tabnet' not in os.listdir('models'):
            return None
        
        if self.model_name + '.zip' not in os.listdir(os.path.join('models', 'tabnet')):
            return None

        model_path = os.path.join('models', 'tabnet', self.model_name + '.zip')
        tb_clf = TabNetClassifier(
            seed=42,
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu',
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            lambda_sparse=0, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type='entmax',
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=10,
        )
        tb_clf.load_model(model_path)
        return tb_clf
    
    def predict(self, X_test):
        model = self.load_model()
        if not model:
            return None
        
        return np.array(model.predict(X_test))
    
    def predict_proba(self, X_test):
        model = self.load_model()
        if not model:
            return None
        
        return np.array(model.predict_proba(X_test))

    def train(self, X_train, Y_train, X_val, Y_val):
        model = self.load_model()
        if model is not None:
            return nn.functional.cross_entropy(torch.tensor(np.array(model.predict_proba(X_val))), torch.tensor(Y_val))

        tb_clf = TabNetClassifier(
            seed=42,
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu',
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            lambda_sparse=0, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type='entmax',
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=10,
        )

        tb_clf.fit(X_train,Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['logloss', 'accuracy'],
            max_epochs=self.max_epochs,
            patience=int(self.max_epochs/10), batch_size=1024, virtual_batch_size=128,
            drop_last=False
        )

        self.save_model(tb_clf)
        
        return tb_clf.best_cost
    
class TabNetMultiCfCustom:
    def __init__(self, model_name, max_epochs=100):
        self.model_name = model_name
        self.max_epochs = max_epochs

    def save_model(self, model):
        if 'models' not in os.listdir('.'):
            os.mkdir('models')

        if 'tabnet' not in os.listdir('models'):
            os.mkdir(os.path.join('models', 'tabnet'))

        model_path = os.path.join(os.path.join('models', 'tabnet', self.model_name))
        path = model.save_model(model_path)
        return path
    
    def load_model(self):
        if 'models' not in os.listdir('.') or 'tabnet' not in os.listdir('models'):
            return None
        
        if self.model_name + '.zip' not in os.listdir(os.path.join('models', 'tabnet')):
            return None

        model_path = os.path.join('models', 'tabnet', self.model_name + '.zip')
        tb_clf = TabNetMultiTaskClassifier(
            seed=42,
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu',
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            lambda_sparse=0, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type='entmax',
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=10,
        )
        tb_clf.load_model(model_path)
        return tb_clf
    
    def predict(self, X_test):
        model = self.load_model()
        if not model:
            return None

        preds = model.predict(X_test)
        if len(preds) == 1:
            preds = preds[0]

        return np.array(preds)
    
    def predict_proba(self, X_test):
        model = self.load_model()
        if not model:
            return None
        
        probs = model.predict_proba(X_test)
        if len(probs) == 1:
            probs = probs[0]

        return np.array(probs)
    
    def train(self, X_train, Y_train, X_val, Y_val):
        
        model = self.load_model()
        if model is not None:
            return nn.functional.cross_entropy(torch.tensor(np.array(model.predict_proba(X_val)[0])), torch.tensor(Y_val).flatten())

        tb_clf = TabNetMultiTaskClassifier(
            seed=42,
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu',
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            lambda_sparse=0, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type='entmax',
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=10,
        )
        
        tb_clf.fit(X_train,Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['logloss', 'accuracy'],
            max_epochs=self.max_epochs,
            patience=int(self.max_epochs/10), batch_size=1024, virtual_batch_size=128,
            drop_last=False
        )

        self.save_model(tb_clf)
        
        return tb_clf.best_cost