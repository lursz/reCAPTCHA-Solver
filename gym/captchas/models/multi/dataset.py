from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms


class MultiObjectDataset(Dataset):
    def __read_file(self, file_name: str) -> tuple[Image.Image, torch.Tensor]:
        image: Image.Image = Image.open(file_name).resize((self.image_width, self.image_width))
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

    
    def __make_negative_sample(self, label: torch.Tensor) -> torch.Tensor:
        label = label.clone()
        
        # label[-1] tells whether the sample is positive [1] or negative [0]
        if label[-1] == 0:
            random_class_index = random.randint(0, self.CLASS_COUNT - 1)
            label[random_class_index] = 1
            return label

        label[-1] = 0
        real_class_index = int(torch.argmax(label[:-1]))

        # select random index, ensure the new index is different from the real class index
        new_class_index = random.choice([i for i in range(self.CLASS_COUNT) if i != real_class_index])
        label[new_class_index] = 1
        label[real_class_index] = 0

        return label
    
    def __init__(self, images_dir: str, is_training: bool, class_name_to_index: dict[str, int], image_width: int = 224) -> None:
        super().__init__()
        self.EPSILON: float = 1e-7
        self.CLASS_COUNT: int = 12
        self.image_width: int = image_width
        self.class_name_to_index: dict[str, int] = class_name_to_index
        self.images_dir: str = images_dir
        self.file_names: list[str] = [str(file_name) for file_name in Path(images_dir).rglob("*.[pj][np]g")]
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width)),
                # transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width)),
                transforms.ToTensor()
            ])
        
        self.cache: dict = {}
        
    def __len__(self) -> int:
        return 2*len(self.file_names) # 2x casue we're adding positive and negative samples
    
    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        is_positive = index % 2 == 0
        img_index = index // 2
        
        if img_index in self.cache:
            img, label = self.cache[img_index]
        else:
            file_name = self.file_names[img_index]
            img, label = self.__read_file(file_name)
            self.cache[img_index] = (img, label)
            
        if not is_positive or label[-1] == 0:
            label = self.__make_negative_sample(label)
            
        augmented_img = self.__augment(img)
        
        # plt.imshow(augmented_img.permute(1, 2, 0).numpy())
        # plt.title(label)
        # plt.show()
            
        return augmented_img, label
        