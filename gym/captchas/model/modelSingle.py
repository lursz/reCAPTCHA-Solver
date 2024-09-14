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
        x = torch.cat((x, class_encoded), dim=1) # concatenate class of the object we're looking for
        x = self.fc(x)
        return x


    

def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device, EPOCHS: int, image_datasets):
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
            outputs_sigmoid = torch.sigmoid(outputs)

            incorrects_count = torch.sum((outputs_sigmoid > 0.5) != (targets[:, 12:] > 0.5))
            running_corrects += len(images) * 16 - incorrects_count
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / (len(image_datasets['train']) * 16)
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

                print(labels[:, 12:], outputs)
                
                val_running_loss += val_loss.cpu().item() * inputs.size(0)
                outputs_sigmoid = torch.sigmoid(outputs)

                incorrects_count = torch.sum((outputs_sigmoid > 0.5) != (labels[:, 12:] > 0.5))
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
            labels = f.read().split('\n')
            labels = [list(map(float, label.split(' '))) for label in labels if label]
            class_ids = [int(label[0]) for label in labels]
            bboxes = [label[1:5] for label in labels] # [[centerx, centery, width, height], ...]

        # One-hot encode
        class_tensors = [torch.zeros((self.CLASS_COUNT)) for _ in range(4)]
        for i, class_id in enumerate(class_ids):
            class_tensors[i][class_id] = 1.0    
        
        class_tensors = torch.cat(class_tensors) # merge class tensors into one one
        
        # Normalize bbox
        bbox = torch.tensor(bboxes)
        bbox = bbox / self.image_width
        
        # Select correct tiles
        tile_width = self.image_width // self.ROWS_COUNT
        selected_tiles_tensor = torch.zeros(self.ROWS_COUNT * self.ROWS_COUNT)

        for i, bbox in enumerate(bboxes):
            center_x, center_y, width, height = bbox
            start_row = int(center_y // tile_width)
            start_col = int(center_x // tile_width)
            end_row = int((center_y + height) // tile_width)
            end_col = int((center_x + width) // tile_width)
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    selected_tiles_tensor[row * self.ROWS_COUNT + col] = 1.0
        
        target = torch.cat((class_tensors, selected_tiles_tensor))
        # return image.to(device), target.to(device)
        return image, target
    
    def __fill_cache(self):
        for idx in range(len(self.images)):
            image, target = self.__read_file(idx)
            self.images_tensors_cache[idx] = image
            self.labels_tensors_cache[idx] = target

    def __init__(self, images_dir, labels_dir, image_width: int = 450) -> None:
        self.CLASS_COUNT = 11
        self.ROWS_COUNT = 4
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
