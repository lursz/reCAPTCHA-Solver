# Imports
import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
from torchvision import models, transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from dotenv import load_dotenv
import datetime
from models.model import modelMulti
from models.model.modelMulti import training_loop, device, BaselineModel, TunedModel

print(f"Using device: {device}")


class Trainer:
    def __init__(self) -> None:
        load_dotenv()
        self.NUM_CLASSES = len(self.image_datasets['train'].classes)
        self.EPOCHS = 20
        dataset_directory: str = os.getenv('CAPTCHA_DATASET_DIR')
        self.result_model_directory: str = os.getenv('CAPTCHA_SAVE_MODELS_DIR')
        # Define dataloaders
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(150),
                transforms.CenterCrop(150),
                transforms.ToTensor(),
                # lambda x: x.to(device),
                # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.1),
                # transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(150),
                transforms.CenterCrop(150),
                transforms.ToTensor(),
                # lambda x: x.to(device),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(150),
                transforms.CenterCrop(150),
                transforms.ToTensor(),
                # lambda x: x.to(device),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        self.image_datasets = {x: datasets.ImageFolder(root=dataset_directory + x, transform=self.data_transforms[x])
                               for x in ['train', 'val', 'test']}
        self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=20, shuffle=True) for x in
                            ['train', 'val', 'test']}

    def train_baseline(self) -> None:
        baseline_model = BaselineModel(self.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
        history = training_loop(baseline_model, criterion, optimizer, self.dataloaders, self.image_datasets,
                                self.EPOCHS)

        max_accuracy = np.int32(max(history['val_accuracy']) * 100)
        torch.save(baseline_model.state_dict(),
                   f'{self.result_model_directory}/model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        baseline_model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")

    def train_resnet(self) -> None:
        resnet_model = TunedModel(self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
        # Train model
        history = training_loop(resnet_model, criterion, optimizer, self.dataloaders, self.image_datasets, self.EPOCHS)

        # Save model
        max_accuracy = np.int32(max(history['val_accuracy']) * 100)
        torch.save(resnet_model.state_dict(),
                   f'saved_models/model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        resnet_model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.train_baseline()
    trainer.train_resnet()
