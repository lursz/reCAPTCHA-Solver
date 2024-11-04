
import os
import numpy as np
import torch
from app.imageProcessing.dataTransform import DataTransformMulti, DataTransformSingle
from gym.captchas.models.multi.modelMulti import ModelMulti
from gym.captchas.models.single.modelSingle import ModelSingle

class BaseModelService:
    def __init__(self, label_index, threshold: float = 0.5) -> None:
        self.label_index = label_index
        self.threshold = threshold

    def load_model(self):
        raise NotImplementedError("Subclasses must implement load_model method.")

    def predict(self, list_of_img):
        raise NotImplementedError("Subclasses must implement predict method.")

class MultiModelService(BaseModelService):
    def __init__(self, label_index, threshold=0.5):
        super().__init__(label_index, threshold)
        self.load_model()
        self.data_transform = DataTransformMulti()

    def load_model(self) -> None:
        self.model = ModelMulti(12)
        self.model.load_state_dict(torch.load(os.getenv('CAPTCHA_MODEL_MULTI'), map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, list_of_img: list) -> list[np.ndarray]:
        print("DUAPAAA", type(list_of_img[0]))
        class_tensor: torch.Tensor = torch.zeros(1, 12)
        class_tensor[0, self.label_index] = 1
        tensor_list: list[torch.Tensor] = self.data_transform.pictures_to_tensors(list_of_img)
        list_of_predictions = []
        for tensor in tensor_list:
            img_tensor = tensor.unsqueeze(0)
            pred = self.model(img_tensor, class_tensor)[0]
            should_select = pred.cpu().detach().numpy() > self.threshold
            list_of_predictions.append(should_select)
        return list_of_predictions

class SingleModelService(BaseModelService):
    def __init__(self, label_index, threshold=0.5):
        super().__init__(label_index, threshold)
        self.load_model()
        self.data_transform = DataTransformSingle()

    def load_model(self):
        self.model = ModelSingle(11)
        self.model.load_state_dict(torch.load(os.getenv('CAPTCHA_MODEL_SINGLE'), map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, list_of_img):
        class_tensor = torch.zeros(1, 11)
        class_tensor[0, self.label_index] = 1
        tensor_list = self.data_transform.pictures_to_tensors(list_of_img)
        img_tensor = tensor_list[0].unsqueeze(0)
        pred = self.model(img_tensor, class_tensor)
        list_of_predictions = pred.cpu().detach().numpy()[0] > self.threshold

        # Split image into tiles
        image = list_of_img[0]
        img_height, img_width, _ = image.shape
        tile_height = img_height // 4
        tile_width = img_width // 4

        tiles = []
        for i in range(4):
            for j in range(4):
                tile = image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
                tiles.append(tile)

        return list_of_predictions, tiles