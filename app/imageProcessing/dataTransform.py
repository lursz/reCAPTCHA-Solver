from torchvision import transforms

class DataTransform:
    def __init__(self) -> None:
        self.data_multi_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(150),
            transforms.CenterCrop(150),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def pictures_to_tensors(self, pics: list):
        tensor_list = [self.data_multi_transform(img) for img in pics]
        return tensor_list