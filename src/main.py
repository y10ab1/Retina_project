import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
import glob 
from dataset import Retina_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy
from torcheval.metrics.classification import BinaryRecall
from torcheval.metrics.classification import BinaryPrecision
from torcheval.metrics.classification import BinaryAUPRC
from torcheval.metrics.classification import BinaryAUROC
from torcheval.metrics.functional import binary_auprc
from torcheval.metrics.functional import binary_auroc
from sklearn.metrics import precision_score, recall_score
import sys 
from model import MY_VGG16
import datetime



torch.manual_seed(8868)
n_epochs = 50
best_val_loss = 10000000
criterion = nn.BCELoss()
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
    
    
vgg_model = models.vgg16(pretrained=True)
model = MY_VGG16(vgg_model)

print(model)
print(model.features_9[1])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_data = Retina_Dataset('train')
print(train_data[0])
val_data = Retina_Dataset('val')
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
sample = next(iter(train_loader))
imgs, lbls = sample
print(lbls)


for epoch in range(n_epochs):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.to(device).to(torch.float32)
        inputs = inputs.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs.size())
        outputs = outputs.squeeze(-1).to(torch.float32)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        print('Iteration {}: Loss: {:.4f}'.format(i,loss.item()))
            
        
            
        
            
    # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        target = torch.tensor([])
        pred = torch.tensor([])
        logits = torch.tensor([])
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            labels = labels.to(device)
            inputs = inputs.to(device) 
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
            
        print('Epoch {}:'.format(epoch)) 
        print(target)
        print(pred)
        print(logits)
        metric = BinaryAccuracy()
        print('Accuracy:')
        ba = metric(pred, target).item()
        print(ba)
        print('Specificity')
        print(SP)
        print('Recall')
        rc = recall_score(target.tolist(), pred.tolist())
        print(rc)
        print('Precision')
        pc = precision_score(target.tolist(), pred.tolist())
        print(pc)
        print('AUPRC')
        auprc = binary_auprc(logits, target).item()
        print(auprc)
        print('AUROC')
        auroc = binary_auroc(logits, target).item()
        print(auroc)
        
        

