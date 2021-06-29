# Cassva train

# import libraries
import os
import numpy as np
import pandas as pd
import torch

from sklearn import model_selection, metrics
import cv2
import torch.nn as nn
import optuna

import torchvision
import timm

import random
from tqdm.auto import tqdm
import time
from torch.cuda import amp

from Engine import Engine
from loss_util import *
from data_util import prep_dataloader
from trn_util import *


def get_path(device='b2'):
    if device=='b2':
        data_path = "/data5/inqlee0704/cassava"
        result_path = '/home/inqlee0704/src/cassava/results'
    elif device=='hpc':
        data_path = "/home/i243l699/temp"
        result_path = '/home/i243l699/work/src/cassava/results'
    return data_path, result_path

class CFG: 
    # img_path = '../input/cassava-leaf-disease-classification/train_images'
    root_path, result_path = get_path(device='b2')
    img_path = os.path.join(root_path,'train_images')
    # resnext: train_bs 32 | valid_bs 64
    img_size = 512
    train_bs = 32
    valid_bs = 64
    target_size = 5
    augmentation = 'normal'
    smoothing = 0.05
    t1 = 0.3
    t2 = 1.0
    epoch = 10
    device = 'cuda'
    weight_decay = 0
    optimizer = 'adam'
    loss = 'BiTemperedLoss'
    # ['cross_entropy', 'LabelSmoothing', 'FocalLoss', 'FocalCosineLoss', 
    # 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']
    # FocalLoss params:
    alpha = 0.627
    gamma = 0.693

    scheduler = 'CosineAnnealingWarmRestarts' # ['CosineAnnealingWarmRestarts',]
    T_0=10
    lr = 1e-4
    min_lr=1e-6
    debug = False
    save = False

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Custom Resnext
class ResNextModel(nn.Module):
    def __init__(self,model,num_classes):
        super().__init__()
        self.convnet = model
        n_features = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(n_features, num_classes)    
    
    def forward(self, img):
        outputs = self.convnet(img)
        return outputs

class EffNetModel(nn.Module):
    def __init__(self,model,num_classes):
        super().__init__()
        self.convnet = model
        n_features = self.convnet.classifier.in_features
        self.convnet.classifier = nn.Linear(n_features, num_classes)    
    
    def forward(self, img):
        outputs = self.convnet(img)
        return outputs


def objective(trial):

    CFG()
    params = {
#    "num_downs": trial.suggest_int("num_downs", 2,6),
#    "initial_filter_size":trial.suggest_int("initial_filter_size",16, 64),
#         "dropout": trial.suggest_uniform("dropout",0.1,0.7),
    # "lr": trial.suggest_loguniform("learning_rate",1e-6,1e-3),
    # 'alpha': trial.suggest_uniform('alpha',0,1),
    # 'gamma': trial.suggest_uniform('gamma',0,5),
    't1': trial.suggest_uniform('t1',0,1),
    't2': trial.suggest_uniform('t2',1,5)
}
    # CFG.alpha = params['alpha']
    # CFG.gamma = params['gamma']
    CFG.t1 = params['t1']
    CFG.t2 = params['t2']
    # KFold
    df = pd.read_csv(os.path.join(CFG.root_path,'train.csv'))
    if (CFG.debug):
        # Only use 5% of dataset
        df = df.sample(frac=0.05).reset_index(drop=True)
    KFOLD = 5
    Fold = model_selection.StratifiedKFold(n_splits=KFOLD,shuffle=True,random_state=42)
    for n, (trn_i,val_i) in enumerate(Fold.split(df, df['label'])):
        df.loc[val_i, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    # print(df.groupby(['fold','label']).size())

    DEVICE = CFG.device
    # FOLD_val_acc = []
    # FOLD_val_loss = []
    #
    for k in range(1):
        save_path = os.path.join(CFG.result_path,f'model_resnext50_focalloss_{k}.pth')
        train_loader, valid_loader = prep_dataloader(df,k,CFG)
        # Model
        # ResNext
        MODEL = torchvision.models.resnext50_32x4d(pretrained=True)
        model = ResNextModel(model=MODEL,num_classes=5)

        # EfficientNet
        # MODEL = timm.create_model('tf_efficientnet_b4_ns',pretrained=True)
        # model = EffNetModel(model=MODEL,num_classes=5)

        loss_fn = get_loss(CFG)
        optimizer = get_optimizer(model,CFG)
        scheduler = get_scheduler(optimizer,CFG)
        scaler = amp.GradScaler()
        eng = Engine(model=model.to(DEVICE),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    device=DEVICE,
                    scaler=scaler)

        best_loss = np.inf
        best_acc = 0
        for epoch in range(CFG.epoch):
            train_loss, train_acc = eng.train(train_loader)
            valid_loss, valid_acc = eng.evaluate(valid_loader)
            eng.epoch += 1
            print(f"Epoch: {epoch},trn_loss: {train_loss:.3f}, trn_acc: {train_acc:.3f}")
            print(f"Epoch: {epoch}, val_loss: {valid_loss:.3f}, val_acc: {valid_acc:.3f}")
            if valid_loss < best_loss:
                print(f"best_loss: {best_loss:.3f} --> {valid_loss:.3f} with acc: {valid_acc:.3f}")
                best_loss = valid_loss
                best_acc = valid_acc
                if CFG.save:
                    torch.save(model.state_dict(), save_path)

            # trial.report(valid_loss,epoch)
            trial.report(valid_acc,epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    # return best_loss
    return best_acc

if __name__ == '__main__':
    # study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    start = time.time()
    seed_everything()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    end = time.time()
    print('Elapsed time: ' + str(end-start))