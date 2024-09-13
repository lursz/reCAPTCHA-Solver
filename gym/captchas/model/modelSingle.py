import os
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
import torchsummary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model.modelTools import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModelSingle(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingle, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():  # Freeze ResNet layers
            param.requires_grad = False
        self.resnet.fc = nn.Identity() # remove the final fully connected layer

        self.fc = nn.Sequential(
            nn.Linear(512 + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

    def forward(self, img: torch.Tensor, class_encoded: torch.Tensor):
        x = self.resnet(img)
        x = torch.cat((x, class_encoded), dim=1) # concatenate the object we're looking for
        x = self.fc(x)

        return x


    

def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device, EPOCHS, image_datasets):
    accuracy_history: list = []
    loss_history: list = []
    val_accuracy_history: list = []
    val_loss_history: list = []
    model.to(device)

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training
        for images, targets in tqdm(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images, targets[:, :12])
            loss = criterion(outputs, targets[:, 12:])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(targets[:, 12:], dim=1)).cpu().item()
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / len(image_datasets['train'])
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, labels[:, :12])
                val_loss = criterion(outputs, labels[:, 12:])
                
                val_running_loss += val_loss.cpu().item() * inputs.size(0)
                val_running_corrects += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels[:, 12:], dim=1)).cpu().item()

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
    
    
    
class ObjectDetectionDataset(Dataset):
    def __read_file(self, idx: int) -> tuple[Image.Image, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt'))

        # images
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # labels
        with open(label_path, 'r') as f:
            label = f.readline().split()
            label = list(map(float, label))

            if len(label) == 0:
                class_id = 0
                bbox = torch.zeros(0)
            else:
                class_id = int(label[0]) + 1
                bbox = torch.tensor(label[1:])

        # One-hot encode
        class_label = torch.zeros(12)
        class_label[class_id] = 1

        # Select correct tiles
        rows_count = 4
        tile_width = self.image_width // rows_count

        selected_tiles_tensor = torch.zeros(rows_count * rows_count)

        for i in range(4):
            for j in range(4):
                tile_x1 = i * tile_width
                tile_y1 = j * tile_width

                for bbox_x1, bbox_y1, bbox_x2, bbox_y2 in zip(bbox[::4], bbox[1::4], bbox[2::4], bbox[3::4]):
                    is_x_intersection = (bbox_x1 <= tile_x1 <= bbox_x2) or (bbox_x1 <= tile_x1 + tile_width <= bbox_x2)
                    is_y_intersection = (bbox_y1 <= tile_y1 <= bbox_y2) or (bbox_y1 <= tile_y1 + tile_width <= bbox_y2)

                    if is_x_intersection and is_y_intersection:
                        selected_tiles_tensor[i * rows_count + j] = 1.0

        target = torch.cat((class_label, selected_tiles_tensor))

        # return image.to(device), target.to(device)
        return image, target
    
    def __fill_cache(self):
        for idx in range(len(self.images)):
            image, target = self.__read_file(idx)
            self.images_tensors_cache[idx] = image
            self.labels_tensors_cache[idx] = target

    def __init__(self, images_dir, labels_dir, image_width: int = 450) -> None:
        self.image_width = image_width

        self.transform = transforms.Compose([
            transforms.Resize((self.image_width, self.image_width)),
            transforms.ToTensor(),
        ])
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(images_dir)

        self.images_tensors_cache = {}
        self.labels_tensors_cache = {}

        self.__fill_cache()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.images_tensors_cache:
            image, target = self.__read_file(idx)
            self.images_tensors_cache[idx] = image
            self.labels_tensors_cache[idx] = target
        else:
            image = self.images_tensors_cache[idx]
            target = self.labels_tensors_cache[idx]

        return image, target
