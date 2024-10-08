from pathlib import Path
import torch
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .dataset import MultiObjectDataset
from .modelMulti import ModelMulti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_multi(model: ModelMulti, criterion: torch.nn.Module, optimizer: Optimizer, dataloaders: dict[str, DataLoader], image_datasets: dict[str, MultiObjectDataset], EPOCHS: int) -> dict:
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training
        for i, (images, targets) in enumerate(tqdm(dataloaders['train'])):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, targets[:, :model.classes_count]) # targets[one hot encoded class, is_positive]
            loss = criterion(outputs, targets[:, model.classes_count:])
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item() * images.cpu().size(0)

            incorrects_count = torch.sum((outputs > 0.5) != (targets[:, model.classes_count:] > 0.5))
            running_corrects += len(images) - incorrects_count
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / (len(image_datasets['train']))
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        
        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(dataloaders['val'])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, labels[:, :model.classes_count])
                loss = criterion(outputs, labels[:, model.classes_count:])
                
                val_running_loss += loss.cpu().item() * inputs.cpu().size(0)
                
                incorrects_count = torch.sum((outputs > 0.5) != (labels[:, model.classes_count:] > 0.5))
                val_running_corrects += len(inputs) - incorrects_count
                    
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
