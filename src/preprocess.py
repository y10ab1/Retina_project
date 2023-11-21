import numpy as np 
import pandas as pd 
import torch
import torchvision
import glob 
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import equalize as equalize_fn
# from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from crop import RetinalCrop

class Retina_Dataset(Dataset):
    def __init__(self,data_type, filepath=None, anno_file=None, select_green=False, clahe=False, img_size=224):
        self.data_type = data_type
        self.select_green = select_green
        self.clahe = clahe
        self.img_size = img_size
        self.crop = RetinalCrop()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            
            self.crop.transform,
            transforms.Resize((self.img_size)),

            ])
                        
            
        if self.data_type == 'train':
            self.file_list = glob.glob("../dataset/Training_Set/Training_Set/Training/*.png") if filepath is None else filepath
            self.anno_file = '../dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv' if anno_file is None else anno_file
        elif self.data_type == 'val':
            self.file_list = glob.glob("../dataset/Evaluation_Set/Evaluation_Set/Validation/*.png") if filepath is None else filepath
            self.anno_file = '../dataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv' if anno_file is None else anno_file
        else:
            self.file_list = glob.glob("../dataset/Test_Set/Test_Set/Test/*.png") if filepath is None else filepath
            self.anno_file = '../dataset/Test_Set/Test_Set/RFMiD_Testing_Labels.csv' if anno_file is None else anno_file
            
        self.label_list = []
        dn = 0
        total = 0
        df = pd.read_csv(self.anno_file)
        for i in range(len(self.file_list)):
            img_id = int(self.file_list[i].split('/')[-1].split('.')[0])
            b = df.index[df['ID'] == img_id].to_list()
            l = df.iloc[b[0]]['Disease_Risk']
            if l == 1:
                dn += 1
            total += 1
            self.label_list.append(l)
        print('Positive: {}, Total: {}, Ratio: {}'.format(dn,total,dn/total))
        
        
        
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_pil = Image.open(self.file_list[idx])
        
        if self.clahe:
            img_pil = self._clahe(img_pil)
        
        image = self.transform_val(img_pil)
        label = self.label_list[idx]
    
        
        filename = self.file_list[idx].split('/')[-1]
        
        return image, label, filename
    
        
    
    
if __name__ == "__main__":
    
    for split in ['train', 'val', 'test']:
    
        target_dataset = Retina_Dataset(split, img_size=224)
        train_loader = torch.utils.data.DataLoader(target_dataset, batch_size=1, shuffle=True)
        save_dir = f'../processed_dataset/{split}/'
        os.makedirs(save_dir, exist_ok=True)
        for i, (image, label, filename) in enumerate(train_loader):
            # save image
            image = image.squeeze(0)
            image = transforms.ToPILImage()(image)
            image.save(f"{save_dir}{filename[0]}")
            break
