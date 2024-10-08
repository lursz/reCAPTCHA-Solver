import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import albumentations as A

class SingleObjectDataset(Dataset):
    def __read_file(self, idx: int) -> tuple[np.ndarray, list[list[int]], torch.Tensor, list[int]]:
        image_path: str = os.path.join(self.images_dir, self.images[idx])
        label_path: str = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.txt'))

        # labels
        with open(label_path, 'r') as f:
            labels = f.read().split('\n')
            labels = [list(map(float, label.split(' '))) for label in labels if label]
            class_ids = [int(label[0]) for label in labels]
            # take edge points of the cluster of points and turn them into bbox
            bboxes = [np.array([min(label[1::2]), min(label[2::2]), max(label[1::2]), max(label[2::2])]) * self.image_width
                        if len(label) > 5 else np.array([label[1] - label[3] / 2, label[2] - label[4] / 2, label[1] + label[3] / 2, label[2] + label[4] / 2]) * self.image_width for label in labels]
            bboxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in bboxes if x2 - x1 > self.EPSILON and y2 - y1 > self.EPSILON]

        class_ids: list[int] = [] if len(bboxes) == 0 else [class_ids[0]] * len(bboxes) 
        class_tensors: torch.Tensor = torch.zeros((self.CLASS_COUNT))
        if len(class_ids) > 0:
            first_class_id = class_ids[0]
            class_tensors[first_class_id] = 1.0
        
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0
        
        return image, bboxes, class_tensors, class_ids

    def __augment(self, image: np.ndarray, bboxes: list[list[int]], class_ids: list[int], class_tensors: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        # Select correct tiles
        tile_width: float = self.image_width / self.ROWS_COUNT
        selected_tiles_tensor: torch.Tensor = torch.zeros(self.ROWS_COUNT * self.ROWS_COUNT)

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
        
        target: torch.Tensor = torch.cat((class_tensors, selected_tiles_tensor))
        
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
    
    def __init__(self, images_dir: str, labels_dir: str, is_training: bool, image_width: int = 450) -> None:
        super().__init__()
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

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> tuple[np.ndarray, torch.Tensor]:
        if index not in self.images_cache:
            image, bboxes, class_tensors, class_ids = self.__read_file(index)
            self.images_cache[index] = image
            self.bboxes_cache[index] = bboxes
            self.class_tensors_cache[index] = class_tensors
            self.class_ids_cache[index] = class_ids
        else:
            image = self.images_cache[index]
            bboxes = self.bboxes_cache[index]
            class_tensors = self.class_tensors_cache[index]
            class_ids = self.class_ids_cache[index]

        return self.__augment(image, bboxes, class_ids, class_tensors)
