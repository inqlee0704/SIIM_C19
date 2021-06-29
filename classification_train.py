import os
import sys
import time
import torch
sys.path.append('../DL_code')
from dotenv import load_dotenv

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import albumentations as A
from sklearn import model_selection, metrics
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
from tqdm.auto import tqdm
# from adamp import AdamP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
import time
from torch.cuda import amp
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob

from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
# import gdcm
from engine import Classifier

# Config
class CFG: 
    load_dotenv()
    root_path = os.getenv('DATA_PATH')
    resized_path = os.path.join(root_path,'siim-covid19-resized-to-256px-jpg')
    data_path = os.path.join(root_path,'siim-covid19-detection')
    img_size = 512
    train_bs = 64
    valid_bs = 64
    target_size = 4
    epoch = 10
    patience = 5
    device = 'cuda'
    optimizer = 'adam'
    loss = 'cross_entropy'
    # ['cross_entropy', 'LabelSmoothing', 'FocalLoss', 'FocalCosineLoss', 
    # 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']
    scheduler = 'CosineAnnealingWarmRestarts' # ['CosineAnnealingWarmRestarts',]
    T_0=10
    lr = 1e-4
    min_lr=1e-6
    save = True
    save_path = './RESULTS/Resnet50_test.pth'

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

class SIIMData(Dataset):
    def __init__(self, df, is_train=True, augments=None, img_size=CFG.img_size):
        super().__init__()
        self.df = df
        self.is_train = is_train
        self.augments = augments
        self.img_size = img_size
        
    def __getitem__(self, idx):
        img_path = self.df['path'].values[idx]
        img = dicom2array(img_path)
        # Gray to RGB conversion #
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (self.img_size,self.img_size))
        
        # Augments must be albumentations
        if self.augments:
            img = self.augments(image=img)['image']
        # else:
        #     img = torch.tensor(img)
        
        img = img/255.0
        img = np.transpose(img,(2,0,1)).astype(np.float32)
        # img = np.transpose(img,(2,0,1))
        if self.is_train:
            label = self.df['label'].values[idx]
            return {
                'img': torch.tensor(img), 
                'targets': torch.tensor(label)
                }
        
        return img
    
    def __len__(self):
        return len(self.df)

# Model #
class ResNet(nn.Module):
    def __init__(self,model,num_classes):
        super().__init__()
        self.convnet = model
        n_features = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(n_features, num_classes)    
    
    def forward(self, img):
        outputs = self.convnet(img)
        return outputs

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    start = time.time()
    
    seed_everything()
    CFG = CFG()

    print('**********')
    print('Configurations:')
    print(CFG.__dict__)
    print('**********')

    # Data prep #
    # Use image level data for detection task
    # image_level_meta = pd.read_csv(os.path.join(CFG.data_path,'train_image_level.csv'))
    # Use study level data for classification
    df = pd.read_csv(os.path.join(CFG.data_path,'train_study_level.csv'))

    df['label'] = 0
    df.loc[df.loc[:,'Negative for Pneumonia'] == 1,'label'] = 0
    df.loc[df.loc[:,'Typical Appearance'] == 1,'label'] = 1
    df.loc[df.loc[:,'Indeterminate Appearance'] == 1,'label'] = 2
    df.loc[df.loc[:,'Atypical Appearance'] == 1,'label'] = 3

    # remove _study
    df['id'] = df['id'].apply(lambda id: id.replace('_study',''))
    train_path = os.path.join(CFG.data_path,'train/')
    data_paths = []
    for id in df['id']:
        data_paths.append(glob.glob(os.path.join(train_path,id+"/*/*"))[0])
    df['path'] = data_paths
    df.head()

    # K-Fold
    KFOLD = 5
    Fold = model_selection.StratifiedKFold(n_splits=KFOLD,shuffle=True,random_state=42)
    for n, (trn_i,val_i) in enumerate(Fold.split(df, df['label'])):
        df.loc[val_i, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    # print(df.groupby(['fold','label']).size())

    k = 0
    df_train = df[df['fold']!=k].reset_index(drop=True)
    df_valid = df[df['fold']==k].reset_index(drop=True)

    # num_workers = 4 is faster than 8
    transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=[-0.2,0.2],
                                        contrast_limit=[-0.2,0.2],
                                        p=0.5),
                A.ShiftScaleRotate(scale_limit=[-0.1,0.3],
                                    shift_limit=0.1,
                                    rotate_limit=20,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    p=0.5),
                # ToTensorV2(p=1.0)
                ])

    train_dataset = SIIMData(df=df_train,augments=transform)
    # train_dataset = SIIMData(df=df_train)
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=CFG.train_bs,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True
                            )
    # valid_dataset = SIIMData(df=df_valid,augments=transform)
    valid_dataset = SIIMData(df=df_valid)
    valid_loader = DataLoader(
                                valid_dataset,
                                batch_size=CFG.valid_bs,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True
                            )

#     base_model = torchvision.models.resnet18(pretrained=True)
    base_model = torchvision.models.resnet50(pretrained=True)
    model = ResNet(model=base_model,num_classes=CFG.target_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=CFG.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0=CFG.T_0, 
                                                    T_mult=1, 
                                                    eta_min=CFG.min_lr,
                                                    last_epoch=-1)
    scaler = amp.GradScaler()

    Engine = Classifier(model=model.to(CFG.device),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        device=CFG.device,
                        scaler=scaler)

    best_loss = np.inf
    best_acc = 0
    for epoch in range(CFG.epoch):
        train_loss, train_acc = Engine.train(train_loader)
        valid_loss, valid_acc = Engine.evaluate(valid_loader)
        Engine.epoch += 1

        print(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"Epoch: {epoch}, valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
        if valid_loss < best_loss:
            print(f"best_loss: {best_loss:.3f} --> {valid_loss:.3f} with acc: {valid_acc:.3f}")
            best_loss = valid_loss
            best_acc = valid_acc
            if CFG.save:
                torch.save(model.state_dict(), CFG.save_path)
    end = time.time()
    print('**********')
    print(f'BEST loss: {best_loss:.3f} with acc {best_acc:.3f}')
    print('**********')
    print(f'Elapsed time: {end-start}')
