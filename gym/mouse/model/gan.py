import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

    
class Generator(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, preprocess_dim: int, sequence_len: int = 50, num_layers_lstm: int = 6) -> None:
        super(Generator, self).__init__()
        self.num_layers_lstm = num_layers_lstm
        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers=num_layers_lstm)
        self.linear1 = nn.Linear(hidden_dim, preprocess_dim)
        self.activation = nn.ReLU()
        self.linear_click = nn.Linear(preprocess_dim, 1)
        self.linear_position = nn.Linear(preprocess_dim, 2)  # click (should stop), x, y, in respect to target, speed
        self.linear_speed = nn.Linear(preprocess_dim, 1)
        self.sequence_len = sequence_len
        self.activation_click = nn.Sigmoid()
        self.activation_position = nn.Tanh()
        self.activation_speed = nn.ReLU()
        
    def forward(self, x):
        # Adjust input_tensor to have a batch dimension
        input_tensor = torch.zeros((self.sequence_len, 3)).to(device)  # Assuming a single batch for simplicity
        
        hidden_size = 16  # Adjust based on your LSTM's hidden size
        hx = torch.zeros(self.num_layers_lstm, hidden_size).to(device)  # Now hx is 3-D
        cx = torch.zeros(self.num_layers_lstm, hidden_size).to(device)  # Now cx is 3-D
        
        cx[0, :2] = x
        hx[0, :2] = x
        
        lstm_out, _ = self.lstm(input_tensor, (hx, cx))
        
        x = self.linear1(lstm_out)
        x = self.activation(x)
        
        click = self.linear_click(x)
        position = self.linear_position(x)
        speed = self.linear_speed(x)

        click = self.activation_click(click)
        position = self.activation_position(position)
        speed = self.activation_speed(speed)
        
        x = torch.cat((click, position, speed), dim=1)
        return x
            

class Discriminator(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(3, hidden_dim, num_layers=3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[-1, :]
        x = self.dropout(last_output)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
