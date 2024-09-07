import torch
from torch import nn
from torch.utils.data import Dataset

class DataSetsChat(Dataset):
    def __init__(self, x,y):
        self.n_sample = len(x)
        self.x_data = x
        self.y_data = y
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_sample