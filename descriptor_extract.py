from ase.visualize import view
from ase.io import Trajectory
import pandas as pd
from ase.symbols import string2symbols
import numpy as np
import torch

rd_dict = {'Cu': 0.67, 'Ni': 0.71, 'Pt':1.04, 'Ag':0.89, 'Au':1.01, 'In':0.63, 'Sn':0.59}
ads_element = {'C', 'H', 'O'}

def get_index_d(n, index, atoms, exclude): 
    
    distances = np.array(atoms.get_all_distances(mic=True))
    distance = distances[index]
    
    dis_argmin = []
    dis_argsort = np.argsort(distance) 
    
    for i in dis_argsort:
        if len(dis_argmin) < n:
            if atoms[i].symbol not in exclude and i!=index:
                dis_argmin.append(i)
        else: 
            break
    return dis_argmin, distance[dis_argmin]

def get_descriptor_labels(df, ads_symbol):
    
    descriptor1_l =[]
    descriptor2_l =[]
    
    for index, row in df.iterrows():
        
        index1 = []
        d1 = []
        index2 = []
        
        sys = row['sys']
        sys_index = row['system'].split('_')
        atoms = Trajectory('./%s_traj/%s_%s.traj' %(sys, sys_index[0], sys_index[1]))[int(sys_index[2])]
        
        ads_index = eval(row['ads_index'])
        
        for index in ads_index:
            ind, d = get_index_d(n=3, index=index, atoms=atoms, exclude=ads_element)
            if atoms[index].symbol == ads_symbol:
                index1.extend(ind)
                d1.extend(d)
        
        argsort = np.argsort(d1)[:3]
        index1 = np.array(index1)[argsort]
        d1 = np.array(d1)[argsort]
        
        for index in index1:
            ind, _ = get_index_d(n=6, index=index, atoms=atoms, exclude=ads_element)
            index2.append(ind)
        
        distances = np.array(atoms.get_all_distances(mic=True))
        
        descriptor1 = torch.zeros(3,6,3)
        descriptor2 = torch.zeros(3,2)
        
        for k1, i in enumerate(index1):
            for k2, j in enumerate(index2[k1]):
                descriptor1[k1, k2, 0] = rd_dict[atoms[i].symbol]
                descriptor1[k1, k2, 1] = rd_dict[atoms[j].symbol]
                descriptor1[k1, k2, 2] = distances[i][j]
        
        for k1, i in enumerate(index1):
            descriptor2[k1, 0] = rd_dict[atoms[i].symbol]
            descriptor2[k1, 1] = d1[k1]
    

        descriptor1_l.append(descriptor1)
        descriptor2_l.append(descriptor2)
        
    
    labels = torch.tensor(df['f_ele'], dtype=torch.float).unsqueeze(dim=1)

    return torch.stack(descriptor1_l), torch.stack(descriptor2_l), labels