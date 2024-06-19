import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from mouseEngine.model.gan import MouseMovementDataset, Generator, Discriminator

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Hyperparameters
sequence_length = 100  # Adjust based on your data
input_dim = 3  # timestamp, x, y


class GanTraining:
    def __init__(self):
        self.EPOCHS = 10000

    def trainGan(self):
        data = []
        with open('mouseEngine/mouse_data.json') as f:
            data = json.load(f)

        data_dfs = [pd.DataFrame(d)[['x', 'y', 'speed']] for d in data if len(d) > 0]

        data_np = [df.values for df in data_dfs]

        # print([len(d) for d in data_np])

        dataset = MouseMovementDataset(data_np)
        dataloader = DataLoader(dataset, shuffle=True)


        batch_size = 64
        hidden_dim = 16
        generator = Generator(3, hidden_dim, hidden_dim).to(device)
        discriminator = Discriminator(hidden_dim).to(device)

        gan_trainer = GanTraining(generator, discriminator, dataloader, 3, 0.00001)
        gan_trainer.train(self.EPOCHS)

        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')


if __name__ == '__main__':
    trainer = GanTraining()
    trainer.trainGan()