import torch
from torch import nn
from torch import Tensor
from torchvision import models
from ..modelTools import ModelTools



class ModelMulti(nn.Module, ModelTools):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classes_count = num_classes
        
        self.resnet: models.ResNet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters(): # freeze the ResNet layers
            param.requires_grad = False

        self.resnet.fc = nn.Identity() # remove the final fully connected layer
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor):
        x = self.resnet(img)
        x = self.fc(x)
        return x
    
    def unfreeze_last_resnet_layer(self):
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def unfreeze_second_to_last_resnet_layer(self):
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
    


class ModelMultiTwoHead(nn.Module, ModelTools):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classes_count = num_classes
        self.resnet: models.ResNet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters(): # freeze the ResNet layers
            param.requires_grad = False

        self.resnet.fc = nn.Identity() # remove the final fully connected layer
        
        self.head_preprocess = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.head_categorical = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        self.head_binary = nn.Sequential(
            nn.Linear(128 + num_classes, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, class_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.resnet(img)
        x = self.head_preprocess(x)
        x_binary = torch.cat((x, class_encoded), dim=1) # concatenate class of the object we're looking for
        x_binary = self.head_binary(x_binary)
        x_categorical = self.head_categorical(x)
        return x_binary, x_categorical
    
    def unfreeze_last_resnet_layer(self):
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def unfreeze_second_to_last_resnet_layer(self):
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True



class ModelMultiSimple(nn.Module, ModelTools): # 99% accuracy, 93% validation accuracy
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet: models.ResNet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters(): # freeze the ResNet layers
            param.requires_grad = False
        for param in self.resnet.layer4.parameters(): # unfreeze the last layer
            param.requires_grad = True
        self.resnet.fc = nn.Identity()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.model(x)
        return x
    

class ModelMultiBaseline(nn.Module, ModelTools): # 98% accuracy, 85% validation accuracy
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.convolve1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # (3, 150, 150)
        self.activation1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # (64, 75, 75)  
        self.convolve3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.activation3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # (128, 37, 37)
        self.convolve4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.activation4 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.convolve5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.activation5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, 3) # (128, 12, 12)
        self.fc1 = nn.Linear(128*12*12, 128)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x) -> Tensor:
        x = self.convolve1(x)
        x = self.activation1(x)
        x = self.batchnorm1(x)
        x = self.convolve2(x)
        x = self.activation2(x)
        x = self.batchnorm2(x)
        x = self.pool1(x)
        x = self.convolve3(x)
        x = self.activation3(x)
        x = self.pool2(x)
        x = self.convolve4(x)
        x = self.activation4(x)
        x = self.batchnorm3(x)
        x = self.convolve5(x)
        x = self.activation5(x)
        x = self.pool3(x)
        x = x.view(-1, 128*12*12)
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc2(x)
        
        return x
