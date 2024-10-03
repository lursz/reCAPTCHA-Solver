import easyocr
import cv2
from matplotlib import pyplot as plt

class OCR:
    def __init__(self) -> None:
        pass    

    def ocr_from_image(self, image_path: str) -> str:
        """
        Reads words from an image using EasyOCR and returns an array of the extracted words.
        """
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path)
        result = [text[1] for text in result] #return only text and not the bounding box
        print(result)
        return result[1]
        
    def normalize_label(self, label: str) -> str:
        label = label.lower()
        return label.strip()