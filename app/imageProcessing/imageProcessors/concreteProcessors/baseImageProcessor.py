from abc import abstractmethod
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class BaseImageProcessor:
    def __init__(self) -> None:
        self.multiple_pics_mode: bool = True
        self.img_cropped = None
        self.header_img = None
        self.list_of_pics = None
        
    def show_image(img: np.ndarray) -> None:
        plt.imshow(img)
        plt.show()
        
    def further_process_captcha_image(output_folder: str) -> np.ndarray:
        pass
        
    @abstractmethod
    def save_pics() -> None:
        pass