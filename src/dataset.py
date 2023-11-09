import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
import glob 
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch



class Retina_Dataset(Dataset):
    def __init__(self,data_type):
        self.data_type = data_type
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            ])
        if self.data_type == 'train':
            self.file_list = glob.glob("../dataset/Training_Set/Training_Set/Training/*.png")
            self.anno_file = '../dataset/Training_Set/Training_Set/RFMiD_Training_Labels.csv'
        elif self.data_type == 'val':
            self.file_list = glob.glob("../dataset/Evaluation_Set/Evaluation_Set/Validation/*.png")
            self.anno_file = '../dataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv'
        else:
            self.file_list = glob.glob("../dataset/Test_Set/Test_Set/Test/*.png")
            self.anno_file = '../dataset/Test_Set/Test_Set/RFMiD_Testing_Labels.csv'
            
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
        image = self.transform(img_pil)
        label = self.label_list[idx]
        return image, label
    
    
    
    
    