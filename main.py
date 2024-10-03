import ast
from pyautogui import sleep
import torch
from app.guiAgent.guiAgent import GuiAgent
from app.imageProcessing.dataTransform import DataTransform, DataTransformMulti, DataTransformSingle
from app.imageProcessing.imageProcessor import ImageProcessor
from app.imageProcessing.ocr import OCR
from torchvision import transforms
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from gym.captchas.model.modelMulti import TunedModel
from gym.captchas.model.modelSingle import ModelSingle
from gym.mouse.mouseEngine import MouseEngine
load_dotenv()

    
    

def main() -> None:
    # Screenshot processing
    gui_agent = GuiAgent()
    ocr = OCR()
    filename = gui_agent.take_screenshot(os.getenv('SCREENSHOTS_FOLDER'))
    sleep(1)

    
    image_processor = ImageProcessor(filename)
    list_of_img =  image_processor.process_captcha_image(os.getenv('CAPTCHA_PICS_FOLDER'))
    
    header_label = ocr.ocr_from_image(os.getenv('CAPTCHA_HEADER_IMG'))
    header_label = ocr.normalize_label(header_label)
    print("Header label", header_label)
    if image_processor.multiple_pics_mode:
        data_transform = DataTransformMulti()
        tensor_list = data_transform.pictures_to_tensors(list_of_img)
        label_to_index: dict = ast.literal_eval(os.environ["CAPTCHA_OBJECTS_INDEX"])
    else:
        data_transform = DataTransformSingle()
        tensor_list = data_transform.pictures_to_tensors([list_of_img])
        label_to_index: dict = ast.literal_eval(os.environ["CAPTCHA_OBJECTS_SINGLE_INDEX"])
    print(f"header_label {header_label}")
    label_index = label_to_index[header_label]
    print(header_label, label_index)
        

    # ML Model
    if image_processor.multiple_pics_mode:
        model = TunedModel(13)
        model.load_state_dict(torch.load(os.getenv('CAPTCHA_MODEL_MULTI'), map_location=torch.device('cpu')))
        model.eval()
        
        TEMP = 1.0
        THRESHOLD = 0.3
        list_of_predictions = []
        for tensor in tensor_list:
            pred = model(tensor.unsqueeze(0))[0]
            pred = torch.nn.functional.softmax(pred / TEMP).detach().numpy()
            should_select = pred[label_index] > THRESHOLD
            list_of_predictions.append(should_select)    
        print(list_of_predictions)
    else:
        model = ModelSingle(11)
        model.load_state_dict(torch.load(os.getenv('CAPTCHA_MODEL_SINGLE'), map_location=torch.device('cpu')))
        model.eval()
        
        THRESHOLD = 0.5
        
        class_tensor = torch.zeros(1, 11)
        class_tensor[0, label_index] = 1
        img_tensor = tensor_list[0].unsqueeze(0)
        
        pred = model(img_tensor, class_tensor)
        
        list_of_predictions = pred.cpu().detach().numpy()[0] > THRESHOLD   
        print(list_of_predictions)    
        
        image = list_of_img
        list_of_img = []
        img_height, img_width, _ = image.shape
        tile_height = img_height // 4
        tile_width = img_width // 4
        
        for i in range(4):
            for j in range(4):
                tile = image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
                list_of_img.append(tile)
        
    # Mouse 
    mouse = MouseEngine()
    for i, img in enumerate(list_of_img):
        if list_of_predictions[i]:
            # plt.imshow(img)
            # plt.show()
            mouse.move_mouse_all_the_way(gui_agent.locate_on_screen(img))
            sleep(1)
    mouse.move_mouse_all_the_way(gui_agent.locate_on_screen(os.getenv('SUBMIT_ICON')))
    
    

if __name__ == '__main__':
    main()