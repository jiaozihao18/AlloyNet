import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from func import *
import numpy as np

torch.manual_seed(450)

descriptor = torch.load("descriptors.pt")
labels = torch.load("labels.pt")
batch_size = 50     
train_loader, test_loader = split_loader(descriptor, labels, batch_size=batch_size, rate=0.7)
                    
lr = 0.02                                                                  
model = MyModel()

criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr = lr) 

def fit(net, criterion, optimizer, batchdata, epochs):
    for epoch in range(epochs):
        for X, y in batchdata:
            X_list = torch.split(X, 1, dim=1)
            yhat = net.forward(X_list)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

for num_epochs in range(50, 3000, 100):
    fit(net = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        batchdata = train_loader, 
        epochs = num_epochs)

    mse_train = mse_cal(train_loader, model)
    mse_test = mse_cal(test_loader, model)
    mae_train = mae_cal(train_loader, model)
    mae_test = mae_cal(test_loader, model)
    
    print('epochs: %i, mse_train: %.3f, mse_test: %.3f, mae_train: %.3f, mae_test: %.3f' %(num_epochs, 
                                                                                           mse_train,
                                                                                           mse_test,
                                                                                           mae_train,
                                                                                           mae_test))

# torch.save(model.state_dict(), 'my_model.pt')