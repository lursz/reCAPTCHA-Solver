import easyocr
import cv2

class OCR:
    def __init__(self) -> None:
        pass    

    def ocr_from_image(image_path: str, threshold: int = 0.2) -> list[str]:
        """
        Reads words from an image using EasyOCR and returns an array of the extracted words.
        """

        reader = easyocr.Reader(['en'])
        img = cv2.imread(image_path)
        try:
            result = reader.readtext(img)
        except FileNotFoundError:
            raise ValueError(f"Invalid image path: {img}")
        except Exception as e:
            raise ValueError(f"Error reading image: {e}")

        if not result:
            return []

        return [word for word in result]  
