import glob
import random
import os
import pandas as pd

import torch 
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PowerDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.data = pd.read_excel(
            path, # path to the excel file that contains dataset
            skiprows=1, # skipping first row to set header later 
            header=None, # not setting header by explicitly setting it 
            )
        
        # resetting column titles from numbers to 'key' values
        # self.data.rename(columns={
        #         0:'frequecy',
        #         1:'bat_capacity',
        #         2:'sf_max',
        #         3:'sf_min',
        #         4:'sf_avg',
        #         5:'tx_pow_max',
        #         6:'tx_pow_min',
        #         7:'sleep_cur',
        #         8:'sensor_time_max',
        #         9:'sensor_time_min',
        #         10:'duty_cycle',
        #         11:'tx_cycle',
        #         12:'avg_bat',
        #     }, 
        #     inplace=True)
        # splitting data into 80/20 for test and train set 
        splitter = int(len(self.data)*0.8) 
        if mode == 'train':
            self.data = self.data.loc[:splitter]
        elif mode == 'test': 
            self.data = self.data.loc[splitter+1:]
        # print('x',self.data.iloc[:,:15])
        # print('y',self.data.iloc[:,15:])

    def __getitem__(self, index):
        x_val = torch.tensor(self.data.iloc[index,:15].values)
        y_val = torch.tensor(self.data.iloc[index,15:].values)

        # Scaling it from [0;1]
        x_val = x_val / x_val.sum(0).expand_as(x_val)
        y_val = y_val / y_val.sum(0).expand_as(y_val)

        return {'x_val': x_val, 'y_val': y_val}

    def __len__(self):
        """returning length of dataset"""
        return len(self.data)

# data = PowerDataset(path='Data_set_no_formula.xlsx')
