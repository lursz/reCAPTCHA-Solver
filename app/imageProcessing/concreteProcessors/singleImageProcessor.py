from .baseImageProcessor import BaseImageProcessor
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class SingleImageProcessor(BaseImageProcessor):
    def __init__(self, img_cropped: np.ndarray, header_img: np.ndarray, list_of_pics: list[np.ndarray]) -> None:
        super().__init__( img_cropped, header_img, list_of_pics)
        self.multiple_pics_mode: bool = False
        self.merged_picture = None
        
    def further_process_captcha_image(self, output_folder: str) -> list[np.ndarray]:
        self.multiple_pics_mode = False
        self.merge_pics()
        self.save_pics(output_folder)
        return [self.merged_picture]
    
    def merge_pics(self) -> None:
        # Merge the pieces to form a single image, remember thay are in 4x4 grid
        tile_height, tile_width, _ = self.list_of_pics[0].shape
        
        self.merged_picture = np.zeros((tile_height * 4, tile_width * 4, 3), dtype=np.uint8)
        for i, pic in enumerate(self.list_of_pics):
            row = i // 4
            col = i % 4
            start_y = row * tile_height
            end_y = (row + 1) * tile_height
            start_x = col * tile_width
            end_x = (col + 1) * tile_width
            self.merged_picture[start_y:end_y, start_x:end_x] = pic

        # show the pic using cv2
        # plt.imshow(self.merged_picture)
        # plt.show()
        
    def save_pics(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}/header.png", self.header_img)
        
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}/merged.png", self.merged_picture)
        

