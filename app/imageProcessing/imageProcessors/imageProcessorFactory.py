import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from .concreteProcessors.baseImageProcessor import BaseImageProcessor
from .initialImageProcessor import InitialImageProcessor
from .concreteProcessors.multiImageProcessor import MultiimageProcessor
from .concreteProcessors.singleImageProcessor import SingleImageProcessor


class ImageProcessorFactory:
    def __init__(self, path: str) -> None:
        self.image_processor = InitialImageProcessor(path)
        
    def get_processor(self, output_folder: str) -> BaseImageProcessor:
        self.image_processor.crop_image_to_captcha()
        self.image_processor.cut_captcha_pics()
        self.image_processor.polishing_the_pics()
        
        if self.image_processor.multiple_pics_mode:
            return self.return_multi_processor()
        else:
            return self.return_single_processor()
        
    def return_multi_processor(self) -> MultiimageProcessor:
        processor = MultiimageProcessor(img_cropped=self.image_processor.img_cropped, 
                                        header_img=self.image_processor.header_img, 
                                        list_of_pics=self.image_processor.list_of_pics)
        return processor
    
    
    def return_single_processor(self) -> SingleImageProcessor:
        processor = SingleImageProcessor(img_cropped=self.image_processor.img_cropped, 
                                         header_img=self.image_processor.header_img, 
                                         list_of_pics=self.image_processor.list_of_pics)
        return processor