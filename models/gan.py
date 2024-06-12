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

class MouseMovementDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.float32)

    
class Generator(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, preprocess_dim: int, sequence_len: int = 50, num_layers_lstm: int = 6) -> None:
        super(Generator, self).__init__()
        self.num_layers_lstm = num_layers_lstm
        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers=num_layers_lstm)
        self.linear1 = nn.Linear(hidden_dim, preprocess_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(preprocess_dim, preprocess_dim)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(preprocess_dim, preprocess_dim)
        self.activation3 = nn.ReLU()
        self.linear_click = nn.Linear(preprocess_dim, 1)
        self.linear_position = nn.Linear(preprocess_dim, 2)  # click (should stop), x, y, in respect to target, speed
        self.linear_speed = nn.Linear(preprocess_dim, 1)
        self.sequence_len = sequence_len
        self.activation_click = nn.Sigmoid()
        self.activation_position = nn.Tanh()
        self.activation_speed = nn.ReLU()
        
    def forward(self, x):
        # Adjust input_tensor to have a batch dimension
        input_tensor = torch.zeros((self.sequence_len, 3))  # Assuming a single batch for simplicity
        
        hidden_size = 16  # Adjust based on your LSTM's hidden size
        hx = torch.zeros(self.num_layers_lstm, hidden_size)  # Now hx is 3-D
        cx = torch.zeros(self.num_layers_lstm, hidden_size)  # Now cx is 3-D
        
        hx[0, :2] = x
        
        lstm_out, _ = self.lstm(input_tensor, (hx, cx))
        
        # print(lstm_out.cpu().detach().numpy()[:, 0], 'generator lstm')
        
        x = self.linear1(lstm_out)
        x = self.activation(x)
        
        x = self.linear2(x)
        x = self.activation2(x)
        
        x = self.linear3(x)
        x = self.activation3(x)
        
        click = self.linear_click(x)
        position = self.linear_position(x)
        speed = self.linear_speed(x)
        
        click = self.activation_click(click)
        position = self.activation_position(position)
        speed = self.activation_speed(speed)
        
        x = torch.cat((click, position, speed), dim=1)
        
        # print(x.shape, 'generator')
        
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
        # print(x.shape, 'discriminator')
        lstm_out, _ = self.lstm(x)
        # print(lstm_out.shape, 'discriminator lstm')
        last_output = lstm_out[-1, :]
        
        # print(last_output.shape)
        
        x = self.dropout(last_output)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        # print(x.shape)
        
        return x

class GanTrainer:
    def __get_click_index(self, fake_prediction_click: torch.Tensor) -> int:
        for i in range(fake_prediction_click.shape[0]):
            if fake_prediction_click[i] > 0.5:
                return i
            
        return len(fake_prediction_click) - 1
    
    def __init__(self, generator, discriminator, dataloader, noise_dim, learning_rate) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.noise_dim = noise_dim
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate*0.5)
        
        self.MAX_STEPS = 1000
    
    def train_discriminator_step(self, real_sequence: torch.Tensor):
        position = torch.randn(2)
        
        self.optimizer_d.zero_grad()
        
        fake_sequence = self.generator(position)
        
        # print(fake_sequence.shape, 'fake sequence before')
        
        # Cut sequence when click is detected
        # index = self.__get_click_index(fake_sequence[:, 0])
        # fake_sequence_cut = fake_sequence[:index + 1, 1:]
        
        # print(real_sequence.shape, 'real sequence')
        # print(fake_sequence_cut.shape, 'fake sequence ', index.item())
        
        # print(real_sequence)
        # print('---')
        # print(fake_sequence)
        # print('---')
        
        real_sequence_prediction = self.discriminator(real_sequence)
        fake_sequence_prediction = self.discriminator(fake_sequence[:, 1:])
        
        one = torch.ones(1) - 0.1 * torch.rand(1)
        zero = 0.1 * torch.rand(1)
        
        one = torch.clamp(one, 0.0, 1.0)
        zero = torch.clamp(zero, 0.0, 1.0)
        
        # print(real_sequence_prediction.shape, fake_sequence_prediction.shape, one.shape, zero.shape)
        
        real_loss = self.criterion(real_sequence_prediction, one)
        fake_loss = self.criterion(fake_sequence_prediction, zero)
                
        loss = real_loss + fake_loss
        
        loss.backward()
        self.optimizer_d.step()
        
        correct_real = torch.sum(real_sequence_prediction > 0.5).item()
        correct_fake = torch.sum(fake_sequence_prediction < 0.5).item()
        
        total_real = real_sequence_prediction.shape[0]
        total_fake = fake_sequence_prediction.shape[0]
        
        return loss.item(), (correct_real + correct_fake) / (total_real + total_fake)
    
    def train_generator_step(self):
        position = torch.randn(2)
        
        self.optimizer_g.zero_grad()
        
        fake_sequence = self.generator(position)
        
        # Cut sequence when click is detected
        # index = self.__get_click_index(fake_sequence[:, 0])
        # fake_sequence_cut = fake_sequence[:index + 1, 1:]
        
        fake_sequence_prediction = self.discriminator(fake_sequence[:, 1:])
        one = torch.ones(1) - 0.1 * torch.rand(1)
        one = torch.clamp(one, 0.0, 1.0)
        
        first_position_loss = torch.square(fake_sequence[0, 1:3] - position).mean()
        last_position_loss = torch.square(fake_sequence[-1, 1:3]).mean()
        
        loss = self.criterion(fake_sequence_prediction, one) + first_position_loss + last_position_loss
        
        loss.backward()
        self.optimizer_g.step()
        
        return loss.item()
    
    def train(self, num_epochs) -> None:
        N = 1
        
        for epoch in range(num_epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            d_acc_total = 0.0
            
            for real_sequence_batch in self.dataloader: # make sure that the format for real_sequences is (sequence_length, 3) for (velocity, x, y) and x, y are relative to target
                real_sequence = real_sequence_batch[0]
                d_loss, d_acc = self.train_discriminator_step(real_sequence)
                d_loss_total += d_loss
                d_acc_total += d_acc
                
                for _ in range(N):
                    g_loss_total += self.train_generator_step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss_total/len(self.dataloader):.4f}, Loss G: {g_loss_total/(len(self.dataloader) * N):.4f}, Accuracy D: {d_acc_total/len(self.dataloader):.4f}")


data = []
with open('mouseEngine/mouse_data.json') as f:
    data = json.load(f)
    
data_dfs = [pd.DataFrame(d)[['x', 'y', 'speed']] for d in data]

data_np = np.array([df.values for df in data_dfs])

print([len(d) for d in data_np])
    
dataset = MouseMovementDataset(data_np)
dataloader = DataLoader(dataset, shuffle=True)

EPOCHS = 1000
batch_size = 64
hidden_dim = 16
generator = Generator(3, hidden_dim, hidden_dim)
discriminator = Discriminator(hidden_dim)

gan_trainer = GanTrainer(generator, discriminator, dataloader, 3, 0.00001)
gan_trainer.train(EPOCHS)
