import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split

class GenData(Dataset):
    def __init__(self, features1, features2, labels):           
        self.features1 = features1                             
        self.features2 = features2
        self.labels = labels                                   
        self.lens = len(features1)                             

    def __getitem__(self, index):     
        return self.features1[index,:], self.features2[index,:], self.labels[index]

    def __len__(self):   
        return self.lens

def split_loader(features1, features2, labels, batch_size=10, rate=0.7):
    data = GenData(features1, features2, labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def mse_cal(data_loader, net):
    data = data_loader.dataset                
    X1 = data[:][0]                            
    X2 = data[:][1]
    y = data[:][2]                            
    yhat = net(X1, X2)
    return F.mse_loss(yhat, y).detach().numpy().round(4)

def mae_cal(data_loader, net):
    data = data_loader.dataset                
    X1 = data[:][0]                            
    X2 = data[:][1]
    y = data[:][2]                           
    yhat = net(X1, X2)
    mae = nn.L1Loss()
    return mae(yhat, y).detach().numpy().round(4)

class FCLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, num_fc, 
                 act_fun=nn.Softplus(), BN_model=None, momentum=0.1):
        super().__init__()
        
        dim_list = [in_dim, *mid_dim*np.ones(num_fc-1, dtype=int)]
        
        layers = []
        for i in range(len(dim_list)-1):
            if BN_model == None:
                layers.extend([nn.Linear(dim_list[i], dim_list[i+1]), act_fun]) 
            elif BN_model == 'pre':
                layers.extend([nn.Linear(dim_list[i], dim_list[i+1]), 
                               nn.BatchNorm1d(dim_list[i+1], momentum=momentum), act_fun])
            elif BN_model == 'post':
                layers.extend([nn.Linear(dim_list[i], dim_list[i+1]), 
                               act_fun, nn.BatchNorm1d(dim_list[i+1], momentum=momentum)])
        layers.append(nn.Linear(mid_dim, out_dim)) # no act_fun in the last layer
    
        self.fc = nn.Sequential(*layers)
        self.fc.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        
    def forward(self, x):
        return self.fc(x)

class AlloyNet(nn.Module):
    def __init__(self, act_fun=nn.Softplus(), num_fc=3, BN_model=None):
        
        super().__init__()
        
        self.fc1 = FCLayer(in_dim=3, mid_dim=3, out_dim=1, num_fc=num_fc, act_fun=act_fun, BN_model=BN_model)
        self.fc2 = FCLayer(in_dim=3, mid_dim=3, out_dim=1, num_fc=num_fc, act_fun=act_fun, BN_model=BN_model)
        self.fc3 = FCLayer(in_dim=2, mid_dim=3, out_dim=1, num_fc=num_fc, act_fun=act_fun, BN_model=BN_model)
        
        self.ln1 = nn.Linear(2, 1)
        
    def forward(self, x1, x2):
        
        m = x1.reshape(x1.shape[0]*3*6, 3)
        m = self.fc1(m)
        m = m.reshape(x1.shape[0], 3, 6)
        m = torch.sum(m, 2)
        m = self.fc2(m)
        
        n = x2.reshape(x2.shape[0]*3, 2)
        n = self.fc3(n)
        n = n.reshape(x2.shape[0], 3)
        n = torch.sum(n, 1, keepdim=True)
        
        mn = torch.cat([m, n], 1)
        
        out = self.ln1(mn)
        
        return out

