# Imports
import ast
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import os
import datetime
from model.modelMulti import ModelMultiBaseline, ModelMultiResnet, ModelMultiSimple, MultiObjectDataset, train_multi
from model.modelSingle import ModelSingle, ObjectDetectionDataset, train_single

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
CAPTCHA_DATASET_DIR: str= 'input/images/'
CAPTCHA_RESULT_MODELS_DIR: str= 'results/'
FREEZED_EPOCHS = 10
UNFREEZED_LAST_LAYER_EPOCHS = 1
EPOCHS = 1


class TrainerMulti:
    def __init__(self) -> None:
        captcha_objects_index = {
            'bicycle': 0,
            'bridge': 1,
            'bus': 2,
            'car': 3,
            'chimney': 4,
            'crosswalk': 5,
            'hydrant': 6,
            'motorcycle': 7,
            'mountain': 8,
            'other': 9,
            'palm': 10,
            'stair': 11,
            'trafficlight': 12
        }
        
        self.datasets: dict[str, MultiObjectDataset] = {
            'train': MultiObjectDataset(CAPTCHA_DATASET_DIR + 'train', True, captcha_objects_index),
            'val': MultiObjectDataset(CAPTCHA_DATASET_DIR + 'val', False, captcha_objects_index),
            'test': MultiObjectDataset(CAPTCHA_DATASET_DIR + 'test', False, captcha_objects_index)
        }
   
        self.dataloaders: dict[str, DataLoader] = {
            'train': DataLoader(self.datasets['train'], batch_size=4, shuffle=True),
            'val': DataLoader(self.datasets['val'], batch_size=4, shuffle=True),
            'test': DataLoader(self.datasets['test'], batch_size=4, shuffle=True)
        }
        self.CLASS_COUNT = self.datasets['train'].CLASS_COUNT
        

    def train(self) -> None:
        model = ModelMultiResnet(self.CLASS_COUNT).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_multi(model, criterion, optimizer, self.dataloaders, self.datasets, FREEZED_EPOCHS)
        model.unfreeze_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train_multi(model, criterion, optimizer, self.dataloaders, self.datasets, UNFREEZED_LAST_LAYER_EPOCHS)
        model.unfreeze_second_to_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train_multi(model, criterion, optimizer, self.dataloaders, self.datasets, EPOCHS)
        max_accuracy = int(max([value.cpu().item() for value in history['val_accuracy']]) * 100)
        torch.save(model.state_dict(), f'{CAPTCHA_RESULT_MODELS_DIR}/model_multi_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")



class TrainerSingle:
    def __init__(self) -> None:
        self.datasets: dict[str, ObjectDetectionDataset] = {
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
        train_single(model, self.dataloaders, criterion, optimizer, FREEZED_EPOCHS, self.datasets)
        model.unfreeze_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train_single(model, self.dataloaders, criterion, optimizer, UNFREEZED_LAST_LAYER_EPOCHS, self.datasets)
        model.unfreeze_second_to_last_resnet_layer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        history = train_single(model, self.dataloaders, criterion, optimizer, EPOCHS, self.datasets)
        max_accuracy = int(max([value.cpu().item() for value in history['val_accuracy']]) * 100)
        torch.save(model.state_dict(), f'{CAPTCHA_RESULT_MODELS_DIR}/model_single_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{max_accuracy}.pt')
        model.plot_accuracy_from_history(history)  #, path="accuracy_plot.png")
        

        

if __name__ == '__main__':
    trainer = TrainerMulti()
    trainer.train()    
    # trainer_single = TrainerSingle()
    # trainer_single.train()
