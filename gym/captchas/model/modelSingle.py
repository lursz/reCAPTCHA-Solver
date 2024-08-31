import os
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
import torchsummary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from gym.captchas.model.modelTools import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModelSingle(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingle, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():  # Freeze ResNet layers
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():  # Unfreeze the last layer
            param.requires_grad = True
        self.resnet.fc = nn.Identity() # remove the final fully connected layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + 4)  # num_classes for class scores + 4 for bbox (x_center, y_center, width, height)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


    

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training
        for images, targets in tqdm(dataloader['train']):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == targets).cpu().item()
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / len(image_datasets['train'])
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader['val']):
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
    
    
    
class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt'))

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load label
        with open(label_path, 'r') as f:
            label = f.readline().split()
            label = list(map(float, label))
            class_id = int(label[0])
            bbox = torch.tensor(label[1:])

        # One-hot encode the class
        class_label = torch.zeros(1)
        class_label[0] = class_id

        # Concatenate class label and bounding box coordinates
        target = torch.cat((class_label, bbox))

        return image, target
class CustomYoloDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self._load_annotations(annotations_file)

    def _load_annotations(self, annotations_file):
        annotations = []
        with open(annotations_file, 'r') as file:
            for line in file:
                data = line.strip().split()
                label = int(data[0])
                coordinates = list(map(float, data[1:]))
                annotations.append((label, coordinates))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label, coordinates = self.annotations[idx]
        img_path = os.path.join(self.img_dir, f'image_{idx}.jpg')  # Modify as needed
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return image, class label, and bounding box coordinates
        return image, label, torch.tensor(coordinates)
    