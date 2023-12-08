import numpy as np 
import pandas as pd 
import torch
import torchvision
import glob 
import cv2

from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import equalize as equalize_fn
# from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2

from crop import RetinalCrop

class Retina_Dataset(Dataset):
    def __init__(self,data_type, filepath=None, anno_file=None, select_green=False, clahe=False, multi_class=False, img_size=600):
        self.data_type = data_type
        self.select_green = select_green
        self.clahe = clahe
        self.crop = RetinalCrop()
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            
            self.crop.transform,
            transforms.Resize((img_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and std for ImageNet

            ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            
            self.crop.transform,
            transforms.Resize((img_size), antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]), # mean and std for ImageNet
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
            
        self.df_labels = pd.read_csv(self.anno_file)
        
        self.multi_class = multi_class
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_pil = Image.open(self.file_list[idx])
        
        if self.clahe:
            img_pil = self._clahe(img_pil)
        
        image = self.transform_train(img_pil) if self.data_type == 'train' else self.transform_val(img_pil)
        
        if self.multi_class:
            # ODC, ODE, ARMD, RS, ODP, TSLN
            label = self.df_labels[['ODC', 'ODE', 'ARMD', 'RS', 'ODP', 'TSLN']]
            label = label.iloc[idx].values  # Select specific row from the label dataframe
            label = torch.tensor(label, dtype=torch.int64)
        else:
            label = self.df_labels['Disease_Risk']
            label = label.iloc[idx]
            label = torch.tensor(label, dtype=torch.int64)
            
        
        if self.select_green:
            image[0,:,:] = 0
            image[2,:,:] = 0
        
        return image, label
    
    def get_label_name(self, label):
        if label.shape[0] == 6:
            label_name = ['ODC', 'ODE', 'ARMD', 'RS', 'ODP', 'TSLN']
        else:
            label_name = ['Disease_Risk']
        return label_name

    
    def _clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        b,g,r = cv2.split(np.array(img))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        image = cv2.merge([b,g,r])
        return Image.fromarray(image)
        
    
    def get_sampler(self):
        # Assuming 'self.df_labels' contains one-hot encoded labels for multilabel classification
        targets = self.df_labels[['ODC', 'ODE', 'ARMD', 'RS', 'ODP', 'TSLN']].values
        

        # Calculate the frequency of each class
        class_sample_count = targets.sum(axis=0)

        # Inverse of class frequency to determine weight
        weight = 1. / np.where(class_sample_count == 0, 1, class_sample_count)  # avoid division by zero

        # Calculate weights for each sample
        samples_weight = np.array([(weight * t).sum() / t.sum() if t.sum() > 0 else 0 for t in targets])

        samples_weight = torch.from_numpy(samples_weight).double()

        # Create a WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler
    
if __name__ == "__main__":
    train_dataset = Retina_Dataset('val', select_green=False, clahe=False, multi_class=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    max_h, max_w = 0, 0
    for i, (image, label) in enumerate(train_loader):
        print(image.shape)
        print(label, label.shape)
        # save image
        # image = image.squeeze(0)
        # image = transforms.ToPILImage()(image)
        # image.save('test.png')
        if i == 10:
            break
    #     max_h = max(max_h, image.shape[2])
    #     max_w = max(max_w, image.shape[3])
        
    # print(max_h, max_w)
    