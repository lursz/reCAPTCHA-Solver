from pathlib import Path
import torch
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .dataset import MultiObjectDataset
from .modelMulti import ModelMulti, ModelMultiTwoHead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_multi(model: ModelMulti, criterion: torch.nn.Module, optimizer: Optimizer, dataloaders: dict[str, DataLoader], image_datasets: dict[str, MultiObjectDataset], EPOCHS: int) -> dict:
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    binary_cross_entropy = torch.nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training
        for images, targets in tqdm(dataloaders['train']):
            images = images.to(device)
            targets = targets.to(device)

            class_encoded = targets[:, :model.classes_count]

            is_any_object = torch.sum(class_encoded, dim=-1, dtype=torch.int32) # sum is equal to 0 only when there is no object class
            class_idx = torch.argmax(class_encoded, dim=-1) * is_any_object + torch.randint_like(is_any_object, model.classes_count) * (1 - is_any_object) # if no class (only 0 in tensor), choose random class

            optimizer.zero_grad()
            outputs = model(images) # targets[one hot encoded class, is_positive]
            
            targets_values = targets[:, model.classes_count:].squeeze()
            targets_gauss = torch.clamp(targets_values + torch.randn_like(targets_values) * 0.1, 0, 1)
            
            batch_indices = torch.arange(class_idx.shape[0]) # for batch 8, indices will be [0, 1, 2, 3, 4, 5, 6, 7]
            selected_values = outputs[batch_indices, class_idx]

            loss = binary_cross_entropy(selected_values, targets_gauss)
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item() * images.cpu().size(0)

            incorrects_count = torch.sum((selected_values > 0.5) != (targets_values > 0.5))
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
            for inputs, labels in tqdm(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                class_encoded = labels[:, :model.classes_count]

                is_any_object = torch.sum(class_encoded, dim=-1, dtype=torch.int32)
                class_idx = torch.argmax(class_encoded, dim=-1) * is_any_object + torch.randint_like(is_any_object, model.classes_count) * (1 - is_any_object)
                batch_indices = torch.arange(class_idx.shape[0])
                
                outputs = model(inputs)
                selected_values = outputs[batch_indices, class_idx]

                targets_values = labels[:, model.classes_count:].squeeze()

                loss = binary_cross_entropy(selected_values, targets_values)
                val_running_loss += loss.cpu().item() * inputs.cpu().size(0)
                
                incorrects_count = torch.sum((selected_values > 0.5) != (targets_values > 0.5))
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


def train_multi_two_head(model: ModelMultiTwoHead, optimizer: Optimizer, dataloaders: dict[str, DataLoader], image_datasets: dict[str, MultiObjectDataset], EPOCHS: int) -> dict:
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    criterion_categorical = torch.nn.CrossEntropyLoss(reduction="none")
    criterion_binary = torch.nn.BCELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training
        for images, targets in tqdm(dataloaders['train']):
            images = images.to(device)
            targets = targets.to(device)
            targets_only_classes = targets[:, :model.classes_count] # one hot encoded class
            targets_class_numbers = torch.argmax(targets_only_classes, dim=1) # class numbers (batch)
            targets_is_positive = targets[:, model.classes_count:]

            optimizer.zero_grad()
            outputs_binary, outputs_categorical = model(images, targets[:, :model.classes_count]) # targets[one hot encoded class, is_positive]
            
            targets_values = targets[:, model.classes_count:]
            targets_gauss = torch.clamp(targets_values + torch.randn_like(targets_values) * 0.1, 0, 1)
            
            loss_binary = criterion_binary(outputs_binary, targets_gauss)

            loss_categorical = (criterion_categorical(outputs_categorical, targets_class_numbers) * targets_is_positive).mean()
            loss = loss_binary + loss_categorical
            
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item() * images.cpu().size(0)

            incorrects_count = torch.sum((outputs_binary > 0.5) != (targets[:, model.classes_count:] > 0.5))
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
            for inputs, labels in tqdm(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs, _ = model(inputs, labels[:, :model.classes_count])
                loss = criterion_binary(outputs, labels[:, model.classes_count:])
                
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




def train_multi_binary(model: ModelMulti, criterion: torch.nn.Module, optimizer: Optimizer, dataloaders: dict[str, DataLoader], image_datasets: dict[str, MultiObjectDataset], EPOCHS: int) -> dict:
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
            
            targets_values = targets[:, model.classes_count:]
            targets_gauss = torch.clamp(targets_values + torch.randn_like(targets_values) * 0.1, 0, 1)
            
            loss = criterion(outputs, targets_gauss)
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