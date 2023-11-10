import numpy as np 
import pandas as pd 
import torch
import torchvision
import glob 
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2 as transforms
# from torchvision import transforms
from sklearn.model_selection import train_test_split


class Retina_Dataset(Dataset):
    def __init__(self,data_type, filepath=None, anno_file=None):
        self.data_type = data_type
        self.transform = transforms.Compose([
            transforms.Resize((224)),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and std for ImageNet

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
        image = self.transform(img_pil)
        label = self.label_list[idx]
        return image, label
    
    
if __name__ == "__main__":
    train_dataset = Retina_Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    max_h, max_w = 0, 0
    for i, (image, label) in enumerate(train_loader):
        print(image.shape)
        print(label)
        # save image
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        image.save('test.png')
        break
    #     max_h = max(max_h, image.shape[2])
    #     max_w = max(max_w, image.shape[3])
        
    # print(max_h, max_w)
    