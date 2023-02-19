import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split

class GenData(Dataset):
    
    def __init__(self, features, labels):           
        self.features = features                   
        self.labels = labels                        
        self.lens = len(features)                  

    def __getitem__(self, index):
        
        return self.features[index,:],self.labels[index]

    def __len__(self):
        
        return self.lens

def split_loader(features, labels, batch_size=10, rate=0.7):
    
    data = GenData(features, labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def mse_cal(data_loader, net):
    
    data = data_loader.dataset                
    X = data[:][0]                            
    y = data[:][1]                           
    
    X_list = torch.split(X, 1, dim=1)         
    yhat = net(X_list)
    
    return F.mse_loss(yhat, y).detach().numpy()

def mae_cal(data_loader, net):
    
    data = data_loader.dataset                
    X = data[:][0]                            
    y = data[:][1]                            
    
    X_list = torch.split(X, 1, dim=1)         
    yhat = net(X_list)
    mae = nn.L1Loss()
    
    return mae(yhat, y).detach().numpy()

def group_cat(x, m, k):
    
    g_list = []
    for i in range(m):
        temp = torch.cat(x[i*k:(i+1)*k], dim=1)
        g_list.append(temp)
        
    return g_list

class BaseModel(nn.Module):
    
    def __init__(self, act_fun=torch.sigmoid, in_features=2, n_hidden=4, 
                 out_features=1, bias=True, BN_model=None,momentum=0.1):
        
        super(BaseModel, self).__init__()
        
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun

    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            out = self.linear2(p1)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            out = self.linear2(p1)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            out = self.linear2(self.normalize1(p1))
            
        return out
    
class ModelBlock(nn.Module):
    
    def __init__(self, k=2, m=18):
        
        self.k = k
        self.m = m
        
        super(ModelBlock, self).__init__()
        
        self.module_list = nn.ModuleList()
        
        for _ in range(self.m):
            self.module_list.append(BaseModel(in_features=self.k, BN_model='pre'))

    def forward(self, input_list): 
        
        x_list = []
        input_list_g = group_cat(input_list, m=self.m, k=self.k) 
        
        for index in range(len(self.module_list)):
            x_list.append(torch.sigmoid(self.module_list[index]
                                        (input_list_g[index])))
        
        #x = torch.cat(x_list, 1)
        
        return x_list 
    
class MyModel(nn.Module):
    
    def __init__(self):
        
        super(MyModel, self).__init__()
        
        self.module_list = nn.ModuleList()
        
        self.block1 = ModelBlock(k=2, m=18)
        self.block2 = ModelBlock(k=6, m=3)
        self.block3 = BaseModel(in_features=3, BN_model='pre')
        
        self.block4 = ModelBlock(k=2, m=3)
        self.block5 = BaseModel(in_features=3, BN_model='pre')
        
        self.block6 = BaseModel(in_features=2, BN_model='pre') 

    def forward(self, input_list): 
        
        x_list = []
        
        x1 = self.block1(input_list[:36])
        x2 = self.block2(x1)
        x2 = torch.cat(x2, dim=1)
        x3 = self.block3(x2)
        #x3 = torch.sigmoid(x3)
        
        x4 = self.block4(input_list[-6:])
        x4 = torch.cat(x4, dim=1)
        x5 = self.block5(x4)
        #x5 = torch.sigmoid(x5)
        
        x6 = torch.cat([x3, x5], 1)
        
        out = self.block6(x6)
        
        return out


