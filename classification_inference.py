import os
import sys
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
from adamp import AdamP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
import time
from torch.cuda import amp
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import gdcm


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
    def __init__(self, df, is_train=False, augments=None, img_size=CFG.img_size):
        super().__init__()
        self.df = df
        self.is_train = is_train
        self.augments = augments
        self.img_size = img_size
        
    def __getitem__(self, idx):
        image_id = self.df['id'].values[idx]
        img_path = self.df['path'].values[idx]
        image = dicom2array(img_path)
        # Gray to RGB conversion #
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (self.img_size,self.img_size))
        
        # Augments must be albumentations
        if self.augments:
            image = self.augments(image=image)['image']
        # else:
        #     image = torch.tensor(image)
        
        image = image/255.0
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        # image = np.transpose(image,(2,0,1))
        if self.is_train:
            label = self.df['label'].values[idx]
            return {
                'img': torch.tensor(image), 
                'targets': torch.tensor(label)
                }
        
        return  {
                'img': torch.tensor(image), 
                }
    
    def __len__(self):
        return len(self.df)

class Engine:
    def __init__(self,model,optimizer,loss_fn,device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.epoch = 0
    
    def infer(self, data_loader):
        self.model.eval()
        preds = []
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        with torch.no_grad():
            for i, batch in pbar:
                inputs = batch['img'].to(self.device,dtype=torch.float)
                outputs = self.model(inputs)
                outputs = outputs.cpu().detach().numpy()
                preds.extend(outputs)
        return np.array(preds)

# Config
class CFG: 
    load_dotenv()
    root_path = os.getenv('DATA_PATH')
    resized_path = os.path.join(root_path,'siim-covid19-resized-to-256px-jpg')
    data_path = os.path.join(root_path,'siim-covid19-detection')
    img_size = 256
    train_bs = 32
    valid_bs = 32
    target_size = 4
    device = 'cuda'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything()
    CFG = CFG()
    df = pd.read_csv(os.path.join(CFG.data_path,'sample_submission.csv'))
    df_study = df[df['id'].str.contains("study")].reset_index(drop=True)
    # df_image = df[df['id'].str.contains("image")].reset_index(drop=True)

    # remove _study
    df_study['id'] = df_study['id'].apply(lambda id: id.replace('_study',''))

    test_path = os.path.join(CFG.data_path,'test/')
    data_paths = []
    for id in df_study['id']:
        data_paths.append(glob.glob(os.path.join(test_path,id+"/*/*"))[0])

    df_study['path'] = data_paths
    test_dataset = SIIMData(df=df_study)
    test_loader = DataLoader(
                                test_dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                # pin_memory=True
                            )

    model_path = './RESULTS/Resnet18_test.pth'
    # Model #
    class ResNet18(nn.Module):
        def __init__(self,model,num_classes):
            super().__init__()
            self.convnet = model
            n_features = self.convnet.fc.in_features
            self.convnet.fc = nn.Linear(n_features, num_classes)    
        
        def forward(self, img):
            outputs = self.convnet(img)
            return outputs

    base_model = torchvision.models.resnet18(pretrained=False)
    model = ResNet18(model=base_model,num_classes=CFG.target_size)

    model.load_state_dict(torch.load(model_path))
    model.to(CFG.device)
    engine = Engine(model=model,optimizer=None,loss_fn=None,device=CFG.device)
    temp_preds = engine.infer(test_loader)
    preds = np.array(F.softmax(torch.from_numpy(temp_preds),dim=1))

    preds_str = []
    for i in range(len(preds)):
        pred_str = f'negative {preds[i,0]} 0 0 1 1 typical {preds[i,1]} 0 0 1 1 indeterminate {preds[i,2]} 0 0 1 1 atypical {preds[i,3]} 0 0 1 1'
        preds_str.append(pred_str)

    df_submit = df[df['id'].str.contains("study")].reset_index(drop=True)
    study_df = pd.DataFrame()
    study_df['id'] = df_submit.id
    study_df['PredictionString'] = preds_str

    study_df.to_csv('./study_submission.csv', index=False)
            