import pandas as pd 
import sys 
import numpy as np 
import glob 
import torch.optim as optim
import torch.nn as nn
import torch
import datetime
import argparse
import os

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

from torch.utils.tensorboard import SummaryWriter
        

def main(args):
    best_val_loss = 10000000
    criterion = nn.BCELoss()

        
        
    # vgg_model = models.vgg16(pretrained=True)
    # model = MY_VGG16(vgg_model)
    # model = VGG16_CBAM(vgg_model)
    
    original_model = models.efficientnet_b7(pretrained=True)
    model = EFF_CBAM(original_model, spatial_attention=True, channel_attention=True) if args.apply_cbam else EFF_CBAM(original_model, spatial_attention=False, channel_attention=False)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    train_data = Retina_Dataset('train', select_green=args.select_green, clahe=args.clahe)
    val_data = Retina_Dataset('val', select_green=args.select_green, clahe=args.clahe)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    sample = next(iter(train_loader))
    imgs, lbls = sample


    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1} of {args.n_epochs}")
        train_loss = 0.0
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
            train_loss += loss.item()
            
        train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
            
                
            
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
                torch.save(model,'./checkpoint/best_model.pt')

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
            
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', metric(pred, target).item(), epoch)
            writer.add_scalar('Val/Specificity', SP, epoch)
            writer.add_scalar('Val/Recall', recall_score(target.tolist(), pred.tolist()), epoch)
            writer.add_scalar('Val/Precision', precision_score(target.tolist(), pred.tolist()), epoch)
            writer.add_scalar('Val/AUPRC', binary_auprc(logits, target).item(), epoch)
            writer.add_scalar('Val/AUROC', binary_auroc(logits, target).item(), epoch)
            
def test(args):
    model = torch.load('./checkpoint/best_model.pt')
    model.to(args.device)
    test_data = Retina_Dataset('test', select_green=args.select_green, clahe=args.clahe)
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
        
        # writer.add_text('test/Accuracy', 'Accuracy: {}'.format(metric(pred, target).item()))
        # writer.add_text('test/Specificity', 'Specificity: {}'.format(SP))
        # writer.add_text('test/Recall', 'Recall: {}'.format(recall_score(target.tolist(), pred.tolist())))
        # writer.add_text('test/Precision', 'Precision: {}'.format(precision_score(target.tolist(), pred.tolist())))
        # writer.add_text('test/AUPRC', 'AUPRC: {}'.format(binary_auprc(logits, target).item()))
        # writer.add_text('test/AUROC', 'AUROC: {}'.format(binary_auroc(logits, target).item()))
        
        writer.add_hparams(
            args.__dict__,
            {'test/accuracy': metric(pred, target).item(),
             'test/specificity': SP,
             'test/recall': recall_score(target.tolist(), pred.tolist()),
             'test/precision': precision_score(target.tolist(), pred.tolist()),
             'test/auprc': binary_auprc(logits, target).item(),
             'test/auroc': binary_auroc(logits, target).item(),
            })
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--apply_cbam', action='store_true')
    parser.add_argument('--select_green', action='store_true')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--log_dir', type=str, default=f'./logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    # writer.add_text('Experiment setting', '\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
    print(args.__dict__)
    exit()
    main(args)
    test(args)
    
    writer.close()