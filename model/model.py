import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
from torchvision import models, transforms, datasets
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class TunedModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
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
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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
