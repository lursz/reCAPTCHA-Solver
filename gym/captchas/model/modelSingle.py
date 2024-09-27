import os
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms
import torchsummary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from model.modelTools import ModelTools
import albumentations as A

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModelSingleWithConv(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingle, self).__init__()
        self.num_classes = num_classes

        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        for param in self.resnet.parameters():  # Freeze ResNet layers
            param.requires_grad = False

        self.avgpool_4x4 = nn.Sequential(
            nn.Conv2d(512 + num_classes, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 16),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor, class_encoded: torch.Tensor):
        x = self.resnet(img)
        class_encoded_reshaped = class_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15, 15)
        x = torch.cat((x, class_encoded_reshaped), dim=1)
        x = self.avgpool_4x4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def unfreeze_last_resnet_layer(self):
        for param in self.resnet[-1].parameters():
            param.requires_grad = True

    def unfreeze_second_to_last_resnet_layer(self):
        for param in self.resnet[-2].parameters():
            param.requires_grad = True
    


class ModelSingle(nn.Module, ModelTools):
    def __init__(self, num_classes: int):
        super(ModelSingle, self).__init__()
        self.num_classes = num_classes

        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():  # Freeze ResNet layers
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
            nn.Linear(128, 16),
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
    
    

def train(model: ModelSingle, dataloader: DataLoader, criterion, optimizer, device, EPOCHS: int, image_datasets):
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
            outputs = model(images, targets[:, :model.num_classes])
            loss = criterion(outputs, targets[:, model.num_classes:])
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item() * images.cpu().size(0)

            incorrects_count = torch.sum((outputs > 0.5) != (targets[:, model.num_classes:] > 0.5))
            running_corrects += len(images) * 16 - incorrects_count
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects / (len(image_datasets['train']) * 16)
        accuracy_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        
        # Validation (only every 5th epoch)
        if epoch % 5 != 4:
            continue
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, labels[:, :model.num_classes])
                val_loss = criterion(outputs, labels[:, model.num_classes:])

                # print(labels[:, model.num_classes:], outputs)
                
                val_running_loss += val_loss.cpu().item() * inputs.cpu().size(0)

                incorrects_count = torch.sum((outputs > 0.5) != (labels[:, model.num_classes:] > 0.5))
                val_running_corrects += len(inputs) * 16 - incorrects_count

        val_epoch_loss = val_running_loss / len(image_datasets['val'])
        val_epoch_acc = val_running_corrects / (len(image_datasets['val']) * 16)

        for _ in range(5):
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
    def __read_file(self, idx: int) -> tuple[Image.Image, list[list[int]], torch.Tensor, list[int]]:
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt'))

        # labels
        with open(label_path, 'r') as f:
            labels = f.read().split('\n')
            labels = [list(map(float, label.split(' '))) for label in labels if label]
            class_ids = [int(label[0]) for label in labels]
            # take edge points of the cluster of points and turn them into bbox
            bboxes = [np.array([min(label[1::2]), min(label[2::2]), max(label[1::2]), max(label[2::2])]) * self.image_width
                        if len(label) > 5 else np.array([label[1] - label[3] / 2, label[2] - label[4] / 2, label[1] + label[3] / 2, label[2] + label[4] / 2]) * self.image_width for label in labels]
            bboxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in bboxes if x2 - x1 > self.EPSILON and y2 - y1 > self.EPSILON]

        class_ids = [] if len(bboxes) == 0 else [class_ids[0]] * len(bboxes) 
        class_tensors = torch.zeros((self.CLASS_COUNT))
        if len(class_ids) > 0:
            first_class_id = class_ids[0]
            class_tensors[first_class_id] = 1.0
        
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0
        
        return image, bboxes, class_tensors, class_ids

    def __augment(self, image: np.ndarray, bboxes: list[list[int]], class_ids: list[int], class_tensors: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        # Select correct tiles
        tile_width = self.image_width / self.ROWS_COUNT
        selected_tiles_tensor = torch.zeros(self.ROWS_COUNT * self.ROWS_COUNT)

        # images
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_ids)
        image = transformed['image']
        # image_copy = image.copy()
        bboxes = transformed['bboxes']
        image = self.pytorch_transform(image)
    
        for bbox in bboxes:
            left_x, top_y, right_x, bottom_y = bbox

            right_x = max(left_x, right_x - 1.0)
            bottom_y = max(top_y, bottom_y - 1.0)

            start_row = int(top_y // tile_width)
            start_col = int(left_x // tile_width)
            end_row = int(bottom_y // tile_width)
            end_col = int(right_x // tile_width)

            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    selected_tiles_tensor[row * self.ROWS_COUNT + col] = 1.0
        
        target = torch.cat((class_tensors, selected_tiles_tensor))
        
        # # visualize bboxes on the picture
        # for bbox in bboxes:
        #     left_x, top_y, right_x, bottom_y = bbox
        #     cv2.rectangle(image_copy, (int(left_x), int(top_y)), (int(right_x), int(bottom_y)), (255, 0, 0), 2)

        # for row in range(self.ROWS_COUNT):
        #     for col in range(self.ROWS_COUNT):
        #         if selected_tiles_tensor[row * self.ROWS_COUNT + col] > 0.5:
        #             cv2.rectangle(image_copy, (int(col * tile_width), int(row * tile_width)), (int((col + 1) * tile_width), int((row + 1) * tile_width)), (0, 255, 0), 2)

        # image_copy = cv2.resize(image_copy, (self.image_width, self.image_width))
        # print(class_ids)
        # if len(class_ids) > 0 and class_ids[0] == 10:
        #     print(self.images[idx])
        #     cv2.imshow('image', image_copy)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows

        return image, target
    
    def __fill_cache(self) -> None:
        for idx in range(len(self.images)):
            image, bboxes, class_tensors, class_ids = self.__read_file(idx)
            self.images_cache[idx] = image
            self.bboxes_cache[idx] = bboxes
            self.class_tensors_cache[idx] = class_tensors
            self.class_ids_cache[idx] = class_ids

    def __init__(self, images_dir: str, labels_dir: str,  is_training: bool, image_width: int = 450) -> None:
        self.EPSILON: float = 1e-7
        self.CLASS_COUNT: int = 11
        self.ROWS_COUNT: int = 4
        self.image_width: int = image_width

        if is_training:
            self.transform = A.Compose([
                    A.Resize(image_width, image_width),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(scale=(0.6, 1.4), shear=(-5, 5), rotate=(-10, 10)),
                    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        else:
            self.transform = A.Compose([
                    A.Resize(image_width, image_width),
                    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        self.pytorch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(images_dir)

        self.images_cache = {}
        self.bboxes_cache = {}
        self.class_tensors_cache = {}
        self.class_ids_cache = {}

        self.__fill_cache()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.images_cache:
            image, bboxes, class_tensors, class_ids = self.__read_file(idx)
            self.images_cache[idx] = image
            self.bboxes_cache[idx] = bboxes
            self.class_tensors_cache[idx] = class_tensors
            self.class_ids_cache[idx] = class_ids
        else:
            image = self.images_cache[idx]
            bboxes = self.bboxes_cache[idx]
            class_tensors = self.class_tensors_cache[idx]
            class_ids = self.class_ids_cache[idx]

        return self.__augment(image, bboxes, class_ids, class_tensors)
