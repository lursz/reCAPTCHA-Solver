import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .dataset import SingleObjectDataset
from .modelSingle import ModelSingle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_single(model: ModelSingle, criterion: torch.nn.Module, optimizer: Optimizer, dataloaders: dict[str, DataLoader], image_datasets: dict[str, SingleObjectDataset], EPOCHS: int) -> dict:
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training
        for images, targets in tqdm(dataloaders['train']):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, targets[:, :model.classes_count]) # targets[one hot encoded class, tiles_to_select]
            loss = criterion(outputs, targets[:, model.classes_count:])
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item() * images.cpu().size(0)

            incorrects_count = torch.sum((outputs > 0.5) != (targets[:, model.classes_count:] > 0.5))
            running_corrects += len(images) * 16 - incorrects_count
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / (len(image_datasets['train']) * 16)
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, labels[:, :model.classes_count])
                val_loss = criterion(outputs, labels[:, model.classes_count:])

                # print(labels[:, model.num_classes:], outputs)
                
                val_running_loss += val_loss.cpu().item() * inputs.cpu().size(0)

                incorrects_count = torch.sum((outputs > 0.5) != (labels[:, model.classes_count:] > 0.5))
                val_running_corrects += len(inputs) * 16 - incorrects_count

        val_epoch_loss = val_running_loss / len(image_datasets['val'])
        val_epoch_acc = val_running_corrects / (len(image_datasets['val']) * 16)

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