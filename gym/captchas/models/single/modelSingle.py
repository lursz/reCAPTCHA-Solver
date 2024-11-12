import torch
from torch import nn
from torchvision import models
from ..modelTools import ModelTools


class ModelSingle(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingle, self).__init__()
        self.classes_count = num_classes

        self.resnet: models.ResNet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters(): # freeze ResNet layers
            param.requires_grad = False
        self.resnet.fc = nn.Identity() # remove the final fully connected layer
        
        self.fc = nn.Sequential(
            nn.Linear(512 + num_classes, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.1),
            nn.Linear(128, 16),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, class_encoded: torch.Tensor):
        x = self.resnet(img)
        x = torch.cat((x, class_encoded), dim=1) # concatenate class of the object we're looking for
        x = self.fc(x)
        return x

    def unfreeze_last_resnet_layer(self):
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def unfreeze_second_to_last_resnet_layer(self):
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
    
    
    
class ModelSingleWithConv(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingleWithConv, self).__init__()
        self.num_classes = num_classes

        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        for param in self.resnet.parameters():  # Freeze ResNet layers
            param.requires_grad = False

        self.avgpool_4x4 = nn.Sequential(
            nn.Conv2d(512 + num_classes, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 16),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor, class_encoded: torch.Tensor):
        x = self.resnet(img)
        class_encoded_reshaped = class_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15, 15)
        x = torch.cat((x, class_encoded_reshaped), dim=1)
        x = self.avgpool_4x4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def unfreeze_last_resnet_layer(self):
        for param in self.resnet[-1].parameters():
            param.requires_grad = True

    def unfreeze_second_to_last_resnet_layer(self):
        for param in self.resnet[-2].parameters():
            param.requires_grad = True