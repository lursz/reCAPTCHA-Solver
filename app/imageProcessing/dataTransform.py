from torch._tensor import Tensor
from torchvision import transforms
from abc import ABC, abstractmethod

class DataTransform(ABC):
    def __init__(self) -> None:
        self.data_transform = transforms.ToTensor()
    
    def pictures_to_tensors(self, pics: list) -> list[Tensor]:
        tensor_list = [self.data_transform(img) for img in pics]
        return tensor_list
    
    
class DataTransformSingle(DataTransform):
    def __init__(self) -> None:
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(450),
            transforms.CenterCrop(450),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        
class DataTransformMulti(DataTransform):
    def __init__(self) -> None:
        self.data_transform = transforms.Compose([
            # lambda img: cv2.medianBlur(img, 3),
            transforms.ToTensor(),
            transforms.Resize(224),
            # transforms.CenterCrop(150)
        ])