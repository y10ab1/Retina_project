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
from torch.utils.data import WeightedRandomSampler

from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelSpecificity
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import MultilabelAUROC

from eff_model import EFF_CBAM
from tqdm import tqdm
torch.manual_seed(8868)

from torch.utils.tensorboard import SummaryWriter
        

def main(args):
    best_val_loss = 10000000
    criterion = nn.BCELoss()

    
    original_model = models.efficientnet_b7(pretrained=True)
    if args.apply_cbam:
        model = EFF_CBAM(original_model, spatial_attention=True, channel_attention=True, num_classes=1 if not args.multi_class else 6)
    else:
        model = EFF_CBAM(original_model, spatial_attention=False, channel_attention=False, num_classes=1 if not args.multi_class else 6)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    train_data = Retina_Dataset('train', select_green=args.select_green, clahe=args.clahe, multi_class=args.multi_class)
    val_data = Retina_Dataset('val', select_green=args.select_green, clahe=args.clahe, multi_class=args.multi_class)
    sampler = train_data.get_sampler() if args.multi_class else None
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, shuffle=False if sampler else True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)


    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1} of {args.n_epochs}")
        train_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for i, data in enumerate(bar, 0):
            inputs, labels = data
            labels = labels.to(args.device).to(torch.float32)
            inputs = inputs.to(args.device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1).to(torch.float32)
            loss = criterion(outputs,labels)
            mAUROC_loss = -MultilabelAUROC(num_labels = 1 if not args.multi_class else 6)(outputs, labels.to(torch.long))
            loss = loss + mAUROC_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            bar.set_description(f"loss: {loss.item():.5f}")
            
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
            bar = tqdm(val_loader)
            for i, data in enumerate(bar, 0):
                inputs, labels = data
                labels = labels.to(args.device)
                inputs = inputs.to(args.device) 
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)
                outputs_label = outputs > 0.5
                logits = torch.cat((logits, outputs.cpu()))
                target = torch.cat((target, labels.cpu())).to(torch.int64)
                pred = torch.cat((pred, outputs_label.cpu()))
                loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
                mAUROC_loss = -MultilabelAUROC(num_labels = 1 if not args.multi_class else 6)(outputs, labels.to(torch.long))
                val_loss += (loss.item() + mAUROC_loss)
                
                bar.set_description(f"loss: {loss.item():.5f}")
                
            dict = {'labels' : target.tolist(), 'logits' : logits.tolist()}
            df = pd.DataFrame(dict)
            df.to_csv('test.csv')
            val_loss = val_loss / len(val_loader)
            print('val_loss: {}'.format(val_loss))
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model,'./checkpoint/best_model.pt')


            specificity = MultilabelSpecificity(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
            Recall = MultilabelRecall(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
            Precision = MultilabelPrecision(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
            acc = MultilabelAccuracy(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
            auroc = MultilabelAUROC(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
            
            print('Accuracy:', acc)
            print('Specificity', specificity)
            print('Recall', Recall)
            print('Precision', Precision)
            print('AUROC', auroc)
            
            print('-----------------------------------------------')
            
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', acc, epoch)
            writer.add_scalar('Val/Specificity', specificity, epoch)
            writer.add_scalar('Val/Recall', Recall, epoch)
            writer.add_scalar('Val/Precision', Precision, epoch)
            writer.add_scalar('Val/AUROC', auroc, epoch)
            
def test(args):
    model = torch.load('./checkpoint/best_model.pt')
    model.to(args.device)
    test_data = Retina_Dataset('test', select_green=args.select_green, clahe=args.clahe, multi_class=args.multi_class)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        target = torch.tensor([])
        pred = torch.tensor([])
        logits = torch.tensor([])
        bar = tqdm(test_loader)
        for i, data in enumerate(bar, 0):
            inputs, labels = data
            labels = labels.to(args.device)
            inputs = inputs.to(args.device) 
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            outputs_label = outputs > 0.5
            logits = torch.cat((logits, outputs.cpu()))
            target = torch.cat((target, labels.cpu())).to(torch.int64)
            pred = torch.cat((pred, outputs_label.cpu()))
            

        dict = {'labels' : target.tolist(), 'logits' : logits.tolist()}
        df = pd.DataFrame(dict)
        df.to_csv('test.csv')
        
        acc = MultilabelAccuracy(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
        recall = MultilabelRecall(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
        precision = MultilabelPrecision(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
        specificity = MultilabelSpecificity(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
        auroc = MultilabelAUROC(num_labels = 1 if not args.multi_class else 6)(pred, target).item()
        
        
        print('Accuracy:', acc)
        print('Specificity', specificity)
        print('Recall', recall)
        print('Precision', precision)
        print('AUROC', auroc)
        
        print('-----------------------------------------------')
        
        writer.add_hparams(
            args.__dict__,
            {'test/accuracy': acc,
             'test/specificity': specificity,
             'test/recall': recall,
             'test/precision': precision,
             'test/auroc': auroc,
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
    parser.add_argument('--multi_class', action='store_true')
    parser.add_argument('--log_dir', type=str, default=f'./logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    print(args.__dict__)
    main(args)
    test(args)
    
    writer.close()