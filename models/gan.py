import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Hyperparameters
sequence_length = 100  # Adjust based on your data
input_dim = 3  # timestamp, x, y
hidden_dim = 256
noise_dim = 100
batch_size = 64
num_epochs = 100
learning_rate = 0.0002

class MouseMovementDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.float32)

# class GeneratorVelocityMethod(nn.Module):
#     def __init__(self, noise_dim: int, hidden_dim: int) -> None:
#         super(GeneratorVelocityMethod, self).__init__()
#         self.linear1 = nn.Linear(noise_dim, hidden_dim)
#         self.activation = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, 2) # returns Vx, Vy
        
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         return x
    
class Generator(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, preprocess_dim: int) -> None:
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(noise_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, preprocess_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(preprocess_dim, 3) # click (should stop), speed, x, y, in respect to target
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        print(lstm_out.shape)

        x = self.linear(lstm_out)
        x = self.activation(x)
        x = self.linear2(x)
        
        return x
            

class Discriminator(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        x = self.flatten(last_output)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x

class GanTrainer:
    def __init__(self, generator, discriminator, dataloader, noise_dim, learning_rate) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.noise_dim = noise_dim
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        self.MAX_STEPS = 1000
    
    def train_discriminator_step(self, real_sequence: torch.Tensor):
        position = torch.randn(2)
        
        self.optimizer_d.zero_grad()
        
        fake_sequence = self.generator(position)
        
        # Cut sequence when click is detected
        index = torch.argmax(fake_sequence[:, 0])
        fake_sequence_cut = fake_sequence[:index][:, 1:]
        
        real_sequence = real_sequence.unsqueeze(0)
        
        real_sequence_prediction = self.discriminator(real_sequence)
        fake_sequence_prediction = self.discriminator(fake_sequence_cut)
        
        one = torch.ones(1) - 0.1 * torch.rand(1)
        zero = 0.1 * torch.rand(1)
        
        one = torch.clamp(one, 0.0, 1.0)
        zero = torch.clamp(zero, 0.0, 1.0)
        
        real_loss = self.criterion(real_sequence_prediction, one)
        fake_loss = self.criterion(fake_sequence_prediction, zero)
        
        loss = real_loss + fake_loss
        
        loss.backward()
        self.optimizer_d.step()
        
        return loss.item()
    
    def train_generator_step(self):
        position = torch.randn(2)
        
        self.optimizer_g.zero_grad()
        
        fake_sequence = self.generator(position)
        
        # Cut sequence when click is detected
        index = torch.argmax(fake_sequence[:, 0])
        fake_sequence_cut = fake_sequence[:index][:, 1:]
        
        fake_sequence_prediction = self.discriminator(fake_sequence_cut)
        one = torch.ones(1) - 0.1 * torch.rand(1)
        one = torch.clamp(one, 0.0, 1.0)
        loss = self.criterion(fake_sequence_prediction, one)
        
        loss.backward()
        self.optimizer_g.step()
        
        return loss.item()
    
    def train(self, num_epochs) -> None:
        for epoch in range(num_epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            
            for real_sequence in self.dataloader: # make sure that the format for real_sequences is (sequence_length, 3) for (velocity, x, y) and x, y are relative to target
                d_loss_total += self.train_discriminator_step(real_sequence)
                g_loss_total += self.train_generator_step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss_total/len(self.dataloader):.4f}, Loss G: {g_loss_total/len(self.dataloader):.4f}")



data = []
with open('mouseEngine/mouse_data.json') as f:
    data = json.load(f)
    
data_dfs = [pd.DataFrame(d)[['x', 'y', 'speed']] for d in data]

# print(data_dfs[0])

data_np = np.array([df.values for df in data_dfs])
    
dataset = MouseMovementDataset(data_np)
dataloader = DataLoader(dataset, shuffle=True)

generator = Generator(noise_dim, hidden_dim, hidden_dim)
discriminator = Discriminator(hidden_dim)

gan_trainer = GanTrainer(generator, discriminator, dataloader, noise_dim, learning_rate)
gan_trainer.train(num_epochs)
