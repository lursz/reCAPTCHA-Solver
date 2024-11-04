import os
import ast
from dotenv import load_dotenv
import numpy as np
from pyautogui import sleep

from app.guiAgent.guiAgent import GuiAgent
from app.imageProcessing.imageProcessors.imageProcessorFactory import ImageProcessorFactory
from app.imageProcessing.ocr import OCR
from gym.mouse.mouseEngine import MouseEngine
from services import MultiModelService, SingleModelService
load_dotenv()

class CaptchaProcessor:
    def __init__(self) -> None:
        self.gui_agent = GuiAgent()
        self.mouse_engine = MouseEngine()
        self.load_environment_variables()

    def load_environment_variables(self) -> None:
        self.screenshots_folder = os.getenv('SCREENSHOTS_FOLDER')
        self.captcha_pics_folder = os.getenv('CAPTCHA_PICS_FOLDER')
        self.captcha_header_img = os.getenv('CAPTCHA_HEADER_IMG')
        self.captcha_objects_index = ast.literal_eval(os.environ["CAPTCHA_OBJECTS_INDEX"])
        self.captcha_objects_single_index = ast.literal_eval(os.environ["CAPTCHA_OBJECTS_SINGLE_INDEX"])
        self.submit_icon = os.getenv('SUBMIT_ICON')
        self.next_icon = os.getenv('NEXT_ICON')
        
    def process_image(self, screenshot_path: str) -> tuple[str, list[np.ndarray], bool]:
        ocr = OCR()
        processor_factory = ImageProcessorFactory(screenshot_path)
        image_processor = processor_factory.get_processor(self.captcha_pics_folder)
        list_of_img = image_processor.further_process_captcha_image(self.captcha_pics_folder)

        # OCR the header label
        header_label = ocr.ocr_from_image(self.captcha_header_img)
        return header_label, list_of_img, image_processor.multiple_pics_mode

    def handle_mouse_actions(self, list_of_img: list[np.ndarray], list_of_predictions: list[np.ndarray]) -> None:
        for i, img in enumerate(list_of_img):
            if list_of_predictions[i]:
                position = self.gui_agent.locate_on_screen(img)
                self.mouse_engine.move_mouse_all_the_way(position)
                
        # consequent captcha
        try:
            next_position = self.gui_agent.locate_on_screen(self.next_icon)
            self.mouse_engine.move_mouse_all_the_way(next_position)
            sleep(1)
            self.process_captcha()
        except Exception as e:
            print("No next icon found.")
                    
        submit_position = self.gui_agent.locate_on_screen(self.submit_icon)
        self.mouse_engine.move_mouse_all_the_way(submit_position)

    def process_captcha(self) -> None:
        # Take screenshot
        screenshot_path = self.gui_agent.take_screenshot(self.screenshots_folder)
        sleep(1)

        # Process image and OCR
        header_label, list_of_img, multiple_pics_mode = self.process_image(screenshot_path)
        print("Header label:", header_label)

        # Get label index and predict
        if multiple_pics_mode:
            label_to_index = self.captcha_objects_index
            label_index = label_to_index[header_label]
            predictor = MultiModelService(label_index)
            list_of_predictions = predictor.predict(list_of_img)
        else:
            label_to_index = self.captcha_objects_single_index
            label_index = label_to_index[header_label]
            predictor = SingleModelService(label_index)
            list_of_predictions, list_of_img = predictor.predict(list_of_img)

        # Perform mouse actions
        self.handle_mouse_actions(list_of_img, list_of_predictions)
        
        
        
        
def main() -> None:
    processor = CaptchaProcessor()
    processor.process_captcha()


if __name__ == '__main__':
    main()