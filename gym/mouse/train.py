import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
from model.trainLoop import GanTrainer
from model.dataset import MouseMovementDataset
from model.gan import Generator, Discriminator

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class GanTraining:
    def __init__(self) -> None:
        self.EPOCHS = 10000
        self.input_file = 'input/mouse_data.json'

    def trainGan(self) -> None:
        data = []
        with open(self.input_file) as f:
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

        gan_trainer = GanTrainer(generator, discriminator, dataloader, 3, 0.00001)
        gan_trainer.train(self.EPOCHS)

        torch.save(generator.state_dict(), 'results/generator.pth')
        torch.save(discriminator.state_dict(), 'results/discriminator.pth')


if __name__ == '__main__':
    trainer = GanTraining()
    trainer.trainGan()