import pandas as pd 
import sys 
import numpy as np 
import glob 
import torch.optim as optim
import torch.nn as nn
import torch
import datetime
import argparse

from dataset import Retina_Dataset
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics.classification import BinaryAccuracy
from torcheval.metrics.classification import BinaryRecall
from torcheval.metrics.classification import BinaryPrecision
from torcheval.metrics.classification import BinaryAUPRC
from torcheval.metrics.classification import BinaryAUROC
from torcheval.metrics.functional import binary_auprc
from torcheval.metrics.functional import binary_auroc
from sklearn.metrics import precision_score, recall_score

from model import MY_VGG16
from vgg_cbam import VGG16_CBAM
from eff_model import EFF_CBAM

torch.manual_seed(8868)

        
        

def main(args):
    best_val_loss = 10000000
    criterion = nn.BCELoss()

        
        
    # vgg_model = models.vgg16(pretrained=True)
    eff_model = models.efficientnet_b7(pretrained=True)
    # model = MY_VGG16(vgg_model)
    # model = VGG16_CBAM(vgg_model)
    model = EFF_CBAM(eff_model)

    print(model)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    train_data = Retina_Dataset('train')
    print(train_data[0])
    val_data = Retina_Dataset('val')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    sample = next(iter(train_loader))
    imgs, lbls = sample
    print(lbls)


    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1} of {args.n_epochs}")
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.to(args.device).to(torch.float32)
            inputs = inputs.to(args.device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.size())
            outputs = outputs.squeeze(-1).to(torch.float32)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            print('Iteration {}: Loss: {:.4f}'.format(i,loss.item()))
                
            
                
            
        print()
        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            target = torch.tensor([])
            pred = torch.tensor([])
            logits = torch.tensor([])
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                labels = labels.to(args.device)
                inputs = inputs.to(args.device) 
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)
                outputs_label = outputs > 0.5
                logits = torch.cat((logits, outputs.cpu()))
                target = torch.cat((target, labels.cpu()))
                pred = torch.cat((pred, outputs_label.cpu()))
                val_loss += criterion(outputs.to(torch.float32), labels.to(torch.float32)).item()
            dict = {'labels' : target.tolist(), 'logits' : logits.tolist()}
            df = pd.DataFrame(dict)
            df.to_csv('test.csv')
            val_loss = val_loss / len(val_loader)
            print('val_loss: {}'.format(val_loss))
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model,'./checkpoint/best_vgg.pt')

            target_list = target.tolist()
            pred_list = pred.tolist()
            FP = 0
            TN = 0
            for z in range(len(pred_list)):
                if target_list[z] == 0 and pred_list[z] == 1:
                    FP += 1
                if target_list[z] == 0 and pred_list[z] == 0:
                    TN += 1
            SP = (TN) / (TN+FP)
            
            metric = BinaryAccuracy()
            print('Accuracy:', metric(pred, target).item())
            print('Specificity', SP)
            print('Recall', recall_score(target.tolist(), pred.tolist()))
            print('Precision', precision_score(target.tolist(), pred.tolist()))
            print('AUPRC', binary_auprc(logits, target).item())
            print('AUROC', binary_auroc(logits, target).item())
            
            print('-----------------------------------------------')
            
def test(args):
    model = torch.load('./checkpoint/best_vgg.pt')
    model.to(args.device)
    test_data = Retina_Dataset('test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        target = torch.tensor([])
        pred = torch.tensor([])
        logits = torch.tensor([])
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            labels = labels.to(args.device)
            inputs = inputs.to(args.device) 
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            outputs_label = outputs > 0.5
            logits = torch.cat((logits, outputs.cpu()))
            target = torch.cat((target, labels.cpu()))
            pred = torch.cat((pred, outputs_label.cpu()))
        dict = {'labels' : target.tolist(), 'logits' : logits.tolist()}
        df = pd.DataFrame(dict)
        df.to_csv('test.csv')
        target_list = target.tolist()
        pred_list = pred.tolist()
        FP = 0
        TN = 0
        for z in range(len(pred_list)):
            if target_list[z] == 0 and pred_list[z] == 1:
                FP += 1
            if target_list[z] == 0 and pred_list[z] == 0:
                TN += 1
        SP = (TN) / (TN+FP)
        
        metric = BinaryAccuracy()
        print('Accuracy:', metric(pred, target).item())
        print('Specificity', SP)
        print('Recall', recall_score(target.tolist(), pred.tolist()))
        print('Precision', precision_score(target.tolist(), pred.tolist()))
        print('AUPRC', binary_auprc(logits, target).item())
        print('AUROC', binary_auroc(logits, target).item())
        
        print('-----------------------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # main(args)
    test(args)