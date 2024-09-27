# Imports
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import os
import datetime
from model.modelMulti import BaselineModel, TunedModel, training_loop
from model.modelSingle import ModelSingle, ObjectDetectionDataset, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
CAPTCHA_DATASET_DIR: str= 'input/images/'
CAPTCHA_RESULT_MODELS_DIR: str= 'results/'
FREEZED_EPOCHS = 20
UNFREEZED_LAST_LAYER_EPOCHS = 50
EPOCHS = 200


class TrainerMulti:
    def __init__(self) -> None:
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

        self.image_datasets = {x: datasets.ImageFolder(root=CAPTCHA_DATASET_DIR + x, transform=self.data_transforms[x]) for x in ['train', 'val', 'test']}
        self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=20, shuffle=True) for x in ['train', 'val', 'test']}
        self.NUM_CLASSES = len(self.image_datasets['train'].classes)
        

    def train_baseline(self) -> None:
        baseline_model = BaselineModel(self.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
        history = training_loop(baseline_model, criterion, optimizer, self.dataloaders, self.image_datasets, EPOCHS)

        max_accuracy = np.int32(max(history['val_accuracy'].cpu()) * 100)
        torch.save(baseline_model.state_dict(), f'{CAPTCHA_RESULT_MODELS_DIR}/model_multi_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        baseline_model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")

    def train_resnet(self) -> None:
        resnet_model = TunedModel(self.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
        # Train model
        history = training_loop(resnet_model, criterion, optimizer, self.dataloaders, self.image_datasets, EPOCHS)
 
        # Save model
        max_accuracy = np.int32(max(history['val_accuracy']) * 100)
        torch.save(resnet_model.state_dict(),
                   f'{CAPTCHA_RESULT_MODELS_DIR}/model_multi_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        resnet_model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")



class TrainerSingle:
    def __init__(self) -> None:
        self.datasets = {
            'train': ObjectDetectionDataset(images_dir='input/yolo/train/images', labels_dir='input/yolo/train/labels', is_training=True),
            'val': ObjectDetectionDataset(images_dir='input/yolo/valid/images', labels_dir='input/yolo/valid/labels', is_training=False),
            'test': ObjectDetectionDataset(images_dir='input/yolo/test/images', labels_dir='input/yolo/test/labels', is_training=False)
        }
        
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], batch_size=4, shuffle=True),
            'val': DataLoader(self.datasets['val'], batch_size=4, shuffle=True),
            'test': DataLoader(self.datasets['test'], batch_size=4, shuffle=True)
        }
        self.NUM_CLASSES = self.datasets['train'].CLASS_COUNT


    def train(self) -> None:
        model = ModelSingle(num_classes=self.NUM_CLASSES)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, self.dataloaders, criterion, optimizer, device, FREEZED_EPOCHS, self.datasets)
        model.unfreeze_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train(model, self.dataloaders, criterion, optimizer, device, UNFREEZED_LAST_LAYER_EPOCHS, self.datasets)
        model.unfreeze_second_to_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train(model, self.dataloaders, criterion, optimizer, device, EPOCHS, self.datasets)
        max_accuracy = int(max([value.cpu().item() for value in history['val_accuracy']]) * 100)
        torch.save(model.state_dict(), f'{CAPTCHA_RESULT_MODELS_DIR}/model_single_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")
        

        

if __name__ == '__main__':
    # trainer = TrainerMulti()
    # trainer.train_baseline()
    # trainer.train_resnet()
    
    trainer_single = TrainerSingle()
    trainer_single.train()
