import torch
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class data(Dataset):
    # construct dataset
    def __init__(self, csvFile, root, transform = None):
        self.root = root
        self.transform = transform
        self.CT = pd.read_csv(csvFile)
    
    # dataset length
    def __len__(self):
        return len(self.CT)
    
    # indexing
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # image
        img_Name = os.path.join(self.root_dir, self.CT.iloc[idx,0])
        img = io.imread(img_Name)
    	cS = self.CT.iloc[idx,3]
    	sample = {'image': img, 'COVID Status': cS}

    	if self.transform:
    		sample = self.transform(sample)
		
    	return sample
