from .baseImageProcessor import BaseImageProcessor
import cv2
import os
import numpy as np


class MultiimageProcessor(BaseImageProcessor):
    def __init__(self, img_cropped: np.ndarray, header_img: np.ndarray, list_of_pics: list[np.ndarray]) -> None:
        super().__init__( img_cropped, header_img, list_of_pics)
        self.multiple_pics_mode: bool = True

    def further_process_captcha_image(self, output_folder: str) -> list[np.ndarray]:
        self.save_pics(output_folder)
        return self.list_of_pics
        
    def save_pics(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            
        cv2.imwrite(f"{path}/header.png", self.header_img)
        for i, pic in enumerate(self.list_of_pics):
            cv2.imwrite(f"{path}/pic_{i}.png", pic)