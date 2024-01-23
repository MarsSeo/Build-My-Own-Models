import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. DataLoad (Can use image dataset such as MNIST)
class DataLoad(Dataset):
    def __init__(self, dataframe, train_or_test = 1):
        self.dataframe = dataframe
        self.train_or_test = train_or_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.train_or_test == 1:
            row = self.dataframe.iloc[idx, 1:].values  
            image_tensor = torch.tensor(row, dtype=torch.float32) / 255.0  
            return image_tensor, image_tensor  
        else:
            row = self.dataframe.iloc[idx, :].values  
            image_tensor = torch.tensor(row, dtype=torch.float32) / 255.0  
            return image_tensor, image_tensor

# 2. AutoEncoder (used modulelist for convenience)
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, 128),
                                             nn.Linear(128, 64),
                                             nn.Linear(64, 32),
                                             nn.Linear(32, 5)])

        self.decoder_layers = nn.ModuleList([nn.Linear(5, 32),
                                            nn.Linear(32, 64),
                                             nn.Linear(64, 128),
                                             nn.Linear(128, input_dim)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = torch.relu(encoder_layer(x))

        for decoder_layer in self.decoder_layers:
            x = torch.relu(decoder_layer(x))
        return torch.sigmoid(x)

# 3. train model (need to define criterion, optimizer)
def train_model(epochs, dataloader, model, criterion, optimizer):
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            inputs, targets = data    
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')
