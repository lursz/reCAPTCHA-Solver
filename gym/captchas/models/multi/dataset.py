from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt


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
        
        def __salt_and_pepper_transform(image: Image.Image, prob_salt: float = 0.01) -> Image.Image:
            image_pixels = np.array(image)
            
            random_values_high = np.random.rand(*image_pixels.shape) < prob_salt * 0.5
            random_values_low = np.random.rand(*image_pixels.shape) < prob_salt * 0.5
            
            low_or_high = np.maximum(random_values_low, random_values_high)
            no_low_or_high = np.logical_not(low_or_high)
            
            processed_image = (image_pixels * no_low_or_high + 255 * random_values_high).astype(np.uint8)
            
            return Image.fromarray(processed_image)
        
        def __cartoonize_image_transform(image: Image.Image) -> Image.Image:
            image_pixels = np.array(image)
            gray_img = cv2.cvtColor(image_pixels, cv2.COLOR_RGB2GRAY)
            
            median_blur = cv2.medianBlur(image_pixels, 3)
            edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            filtered_image = cv2.bilateralFilter(median_blur, 9, 200, 200)
            cartoon_img = cv2.bitwise_and(filtered_image, filtered_image, mask=edges)
            
            return Image.fromarray(cartoon_img)
        
        if is_training:           
            self.transform = transforms.Compose([
                transforms.Resize((self.image_width, self.image_width)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    lambda x: x,
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.05),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                    lambda x: __salt_and_pepper_transform(x, prob_salt=0.005),
                    lambda x: __salt_and_pepper_transform(x, prob_salt=0.01),
                    lambda x: __cartoonize_image_transform(x)
                ], p=[0.3, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1]),
                #transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
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
        # plt.savefig(f"temp/{uuid.uuid4()}")
            
        return augmented_img, label
        
