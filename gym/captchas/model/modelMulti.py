import torch
from torch import nn
from torch import Tensor
from torchvision import models
from tqdm import tqdm
from .modelTools import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TunedModel(nn.Module, ModelTools): # 99% accuracy, 93% validation accuracy
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
    

class BaselineModel(nn.Module, ModelTools): # 98% accuracy, 85% validation accuracy
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
    

def training_loop(model, criterion, optimizer, dataloaders, image_datasets, EPOCHS: int = 1000):
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    for epoch in range(EPOCHS):
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item() * inputs.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels).cpu().item()
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / len(image_datasets['train'])
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                
                # debug print images
                # for input, label, prediction in zip(inputs, labels, torch.argmax(outputs, dim=1)):
                #     plt.imshow(input.permute(1, 2, 0).cpu().numpy())
                #     plt.title(f"Label: {image_datasets['val'].classes[label]}, Prediction: {image_datasets['val'].classes[prediction]}")
                #     plt.show()
                
                val_running_loss += val_loss.cpu().item() * inputs.size(0)
                val_running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels).cpu().item()

        val_epoch_loss = val_running_loss / len(image_datasets['val'])
        val_epoch_acc = val_running_corrects / len(image_datasets['val'])
        val_accuracy_history.append(val_epoch_acc)
        val_loss_history.append(val_epoch_loss)

        print(f'Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
    history: dict = {
            'accuracy': accuracy_history,
            'loss': loss_history,
            'val_accuracy': val_accuracy_history,
            'val_loss': val_loss_history
        }
    
    return history
