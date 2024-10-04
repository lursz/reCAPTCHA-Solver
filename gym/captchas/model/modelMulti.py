from pathlib import Path
import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from torchvision import models, transforms
from .modelTools import ModelTools



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ModelMultiResnet(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super().__init__()
        self.classes_count = num_classes
        
        self.resnet: models.ResNet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters(): # freeze the ResNet layers
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
            nn.Linear(128, out_features=1),
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
    

def train_multi(model, criterion, optimizer, dataloaders, image_datasets, EPOCHS: int = 1000) -> dict:
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training
        for images, targets in tqdm(dataloaders['train']):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, targets[:, :model.classes_count]) # targets[one hot encoded class, is positive]
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
            for inputs, labels in tqdm(dataloaders['val']):
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


class MultiObjectDataset(Dataset):
    def __read_file(self, file_name: str) -> tuple[Image.Image, torch.Tensor]:
        image = Image.open(file_name).resize((self.image_width, self.image_width))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        label = torch.zeros(self.CLASS_COUNT + 1) # +1 cause we're adding positive/negative label
        class_name = file_name.split('/')[-2].lower()
        
        if class_name != "other":
            class_index = self.class_name_to_index[class_name]
            label[class_index] = 1
            label[-1] = 1
        
        return image, label
    
    def __augment(self, image: Image.Image) -> Image.Image:
        return self.transform(image)
    
    def __make_reverse_sample(self, label: torch.Tensor) -> torch.Tensor:
        if label[-1] == 0: # label[-1] tells whether the sample is positive or negative
            random_class_index = random.randint(0, self.CLASS_COUNT)
            label[random_class_index] = 1
            return label
        
        label[-1] = 0
        
        real_class_index = torch.argmax(label[:-1])
        random_number = random.randint(0, self.CLASS_COUNT - 1)
        new_class_index = random_number if random_number < real_class_index else random_number + 1
        label[new_class_index] = 1
        
        return label
    
    def __init__(self, images_dir: str, is_training: bool, class_name_to_index: dict[str, int], image_width: int = 224) -> None:
        super().__init__()
        self.EPSILON: float = 1e-7
        self.CLASS_COUNT: int = 13
        self.image_width: int = image_width
        self.class_name_to_index = class_name_to_index
        self.images_dir = images_dir
        self.file_names = [str(file_name) for file_name in Path(images_dir).rglob("*.[pj][np]g")]
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width)),
                transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width)),
                transforms.ToTensor()
            ])
        
        self.images_dir = images_dir
        self.cache: dict = {}
        
    def __len__(self) -> int:
        return 2*len(self.file_names) # 2x casue we're adding positive and negative samples
    
    def __getitem__(self, index) -> tuple[Image.Image, torch.Tensor]:
        is_positive = index % 2 == 0
        img_index = index // 2
        
        if img_index in self.cache:
            img, label = self.cache[img_index]
        else:
            file_name = self.file_names[img_index]
            img, label = self.__read_file(file_name)
            self.cache[img_index] = (img, label)
            
        if not is_positive:
            label = self.__make_reverse_sample(label)
            
        return self.__augment(img), label
        