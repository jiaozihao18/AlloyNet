import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from func import *
import numpy as np
import pandas as pd

torch.manual_seed(450)

batch_size = 50                              
lr = 0.02                                    
num_epochs = 500                               

descriptor1 = torch.load("descriptors1.pt")
descriptor2 = torch.load("descriptors1.pt")
labels = torch.load("labels.pt")

train_loader, test_loader = split_loader(descriptor1, descriptor2, labels,  
                                         batch_size=batch_size, rate=0.7)

model = AlloyNet()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr = lr) 

def fit(net, criterion, optimizer, batchdata, epochs):
    for epoch in range(epochs):
        for X1, X2, y in batchdata:
            yhat = net.forward(X1, X2)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

mse_train = []
mse_test = []
mae_train = []
mae_test = []

mse_te_min = 100

for num_epochs in range(50, 3000, 100):
    fit(net = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        batchdata = train_loader, 
        epochs = num_epochs)

    mse_tr = mse_cal(train_loader, model)
    mse_te = mse_cal(test_loader, model)
    mse_train.append(mse_tr)          
    mse_test.append(mse_te)           
    
    if mse_te < mse_te_min:
        torch.save(model.state_dict(), 'model_best.pt')
        mse_te_min = mse_te
    
    mae_tr = mae_cal(train_loader, model)
    mae_te = mae_cal(test_loader, model)
    mae_train.append(mae_tr)          
    mae_test.append(mae_te)       
    
    print('epochs: %i, mse_train: %.3f, mse_test: %.3f, mse_test_min: %.3f' 
          %(num_epochs, mse_tr, mse_te, mse_te_min))
    
torch.save(model.state_dict(), 'model.pt')

df_msae = pd.DataFrame({"mse_train": mse_train, "mse_test": mse_test,
                        "mae_train": mae_train, "mae_test": mae_test})
df_msae.to_csv('msae.csv', index=False)