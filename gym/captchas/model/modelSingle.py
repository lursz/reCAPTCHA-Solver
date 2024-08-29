import os
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import torchsummary
from PIL import Image
from gym.captchas.model.modelTools import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ModelYolo(nn.Module, ModelTools):
    def __init__(self):
        super(ModelYolo, self).__init__()
        data_config_path = 'gym/captchas/model/input/yolo/data.yaml' 
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(device)
        
    def forward(self, x):
        return self.model(x)
    

class ModelSingle(nn.Module, ModelTools):
    def __init__(self, num_classes: int) -> None:
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
    
    

if __name__ == "__main__":
    model = ModelYolo()
    torchsummary.summary(model, (3, 640, 640))
    print(model)
