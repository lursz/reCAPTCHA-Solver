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
import visualizeData
import model
from model import training_loop, device
import datetime

print(f"Using device: {device}")



# Define dataloaders
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        # lambda x: x.to(device),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.2, hue=0.1),
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

# Define data directories
print("Loading data...")
data_dir: str = 'processed_dataset/images/'
image_datasets = {x: datasets.ImageFolder(root=data_dir+x, transform=data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=20, shuffle=False) for x in ['train', 'val', 'test']}
print("Data loaded")

num_classes = len(image_datasets['train'].classes)

# Define model
resnet_model = model.TunedModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

# Train model
EPOCHS = 100
history = training_loop(resnet_model, criterion, optimizer, dataloaders, image_datasets, EPOCHS)
# Save model
max_accuracy = np.int32(max(history['val_accuracy']) * 100)
torch.save(resnet_model.state_dict(), f'saved_models/model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')

# Plot training history
visualizeData.plot_accuracy_from_history(history)#, path="accuracy_plot.png")
# save plot to pickle

