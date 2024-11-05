import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.adam import Adam
from .gan import Generator, Discriminator
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')



class GanTrainer:
    def __get_click_index(self, fake_prediction_click: torch.Tensor) -> int:
        for i in range(fake_prediction_click.shape[0]):
            if fake_prediction_click[i] > 0.5:
                return i
            
        return len(fake_prediction_click) - 1
    
    def __init__(self, generator: Generator, discriminator: Discriminator, dataloader: DataLoader, noise_dim: int, learning_rate: float) -> None:
        self.generator: Generator = generator
        self.discriminator: Discriminator = discriminator
        self.dataloader: DataLoader = dataloader
        self.noise_dim: int = noise_dim
        self.criterion = nn.BCELoss()
        self.optimizer_g = Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=learning_rate*0.4)
        
        self.MAX_STEPS = 1000
    
    def train_discriminator_step(self, real_sequence: torch.Tensor) -> tuple:
        position = (torch.rand(2) * 2 - 1).to(device)
        
        self.optimizer_d.zero_grad()
        
        fake_sequence = self.generator(position)
        
        real_sequence_prediction = self.discriminator(real_sequence)
        fake_sequence_prediction = self.discriminator(fake_sequence[:, 1:])
        
        one = (torch.ones(1) - 0.1 * torch.rand(1)).to(device)
        zero = 0.1 * torch.rand(1).to(device)
        
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
        position = (torch.rand(2) * 2 - 1).to(device)
        
        self.optimizer_g.zero_grad()
        
        fake_sequence = self.generator(position)
        
        fake_sequence_prediction = self.discriminator(fake_sequence[:, 1:])
        one = (torch.ones(1) - 0.1 * torch.rand(1)).to(device)
        one = torch.clamp(one, 0.0, 1.0).to(device)
        
        first_position_loss = torch.square(fake_sequence[0, 1:3] - position).mean()
        last_position_loss = torch.square(fake_sequence[-1, 1:3]).mean()
        
        loss = self.criterion(fake_sequence_prediction, one) + first_position_loss + last_position_loss
        
        loss.backward()
        self.optimizer_g.step()
        
        return loss.item()
    
    def print_fake_sequence(self):
        position = (torch.rand(2) * 2 - 1).to(device)
        
        fake_sequence = self.generator(position)
        
        print(f"Starting position: {position.cpu().detach().numpy()}")
        print(fake_sequence.cpu().detach().numpy())
    
    def train(self, num_epochs) -> None:
        N = 1
        
        for epoch in range(num_epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            d_acc_total = 0.0
            gen_train_steps = 0
            
            for real_sequence_batch in self.dataloader: # make sure that the format for real_sequences is (sequence_length, 3) for (velocity, x, y) and x, y are relative to target
                real_sequence = real_sequence_batch[0]
                d_loss, d_acc = self.train_discriminator_step(real_sequence)
                d_loss_total += d_loss
                d_acc_total += d_acc
                
                extra_steps = 2 * (d_acc > 0.96) + 1 * (d_acc > 0.9) + 1 * (d_acc > 0.8)
                
                for _ in range(N + extra_steps):
                    g_loss_total += self.train_generator_step()
                    gen_train_steps += 1
            
            self.print_fake_sequence()
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss_total/len(self.dataloader):.4f}, Loss G: {g_loss_total/gen_train_steps:.4f}, Accuracy D: {d_acc_total/len(self.dataloader):.4f}")
            
            if epoch % 1000 == 0:
                torch.save(self.generator.state_dict(), f'results/generator_{epoch}.pth')
                torch.save(self.discriminator.state_dict(), f'results/discriminator_{epoch}.pth')


