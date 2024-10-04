from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class BaseImageProcessor:
    def __init__(self, img_cropped: np.ndarray, header_img: np.ndarray, list_of_pics: list[np.ndarray]) -> None:
        self.multiple_pics_mode: bool = True
        self.img_cropped = img_cropped
        self.header_img = header_img
        self.list_of_pics = list_of_pics
        
    def show_image(self, img: np.ndarray) -> None:
        plt.imshow(img)
        plt.show()
        
    @abstractmethod
    def further_process_captcha_image(self, output_folder: str) -> list[np.ndarray]:
        pass
        
    @abstractmethod
    def save_pics(self, path: str) -> None:
        pass