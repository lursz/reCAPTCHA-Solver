import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
from torchvision import models, transforms, datasets
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class ModelTuning(nn.Module):
    def __init__(self):
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
            nn.Linear(256, 1),
            nn.Sigmoid(),
            nn.Flatten(0, 1)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.model(x)
        return x
    
    
# model3 = ModelTuning().to(device)
# Define loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.RMSprop(model3.parameters(), lr=2e-5)




def training_loop(model3, criterion, optimizer, dataloaders, image_datasets, EPOCHS: int = 1000):
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    for epoch in range(EPOCHS):
        model3.train()
        
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders['train'], disable=True):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model3(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item() * inputs.size(0)
            running_corrects += torch.sum((outputs > 0.5) == labels.byte())
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / len(image_datasets['train'])
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)

        # Validation loop
        model3.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['validation'], disable=True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model3(inputs)
                val_loss = criterion(outputs, labels.float())
                val_running_loss += val_loss.cpu().item() * inputs.size(0)
                val_running_corrects += torch.sum((outputs > 0.5) == labels.byte())
        val_epoch_loss = val_running_loss / len(image_datasets['validation'])
        val_epoch_acc = val_running_corrects / len(image_datasets['validation'])
        val_accuracy_history.append(val_epoch_acc)
        val_loss_history.append(val_epoch_loss)

        print(f'Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
    history3: dict = {
            'accuracy': accuracy_history,
            'loss': loss_history,
            'val_accuracy': val_accuracy_history,
            'val_loss': val_loss_history
        }
    