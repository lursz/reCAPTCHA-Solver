from pyautogui import sleep
import torch
from guiAgent.guiAgent import GuiAgent
from imageProcessing.imageProcessor import ImageProcessor
from imageProcessing.ocr import OCR
from models.gan import Generator
from models.model import TunedModel
from mouseEngine.mouseEngine import MouseEngine
from torchvision import models, transforms, datasets
from models.train import num_classes




def main() -> None:
    print("CAPTCHA Solver")
    
    ss_path: str = 'guiAgent/screenshots'
    gui_agent = GuiAgent()
    gui_agent.open_browser("localhost")
    sleep(3)
    gui_agent.click_checkbox()
    sleep(3)
    filename = gui_agent.take_screenshot(ss_path)
    # gui_agent.closeTab()
    sleep(3)
    image_processor = ImageProcessor(filename)
    # image_processor.show_image()
    list_of_img =  image_processor.process_captcha_image('imageProcessing/captcha_pics')
    
    # ocr = OCR()
    # list_of_words = ocr.ocr_from_image('imageProcessing/captcha_pics/header.png')
    # header_label = list_of_words[-1]

    # IMAGE MODEL 
    pic_model = TunedModel(num_classes)
    pic_model.load_state_dict('models/captcha_model.pth')

    # apply this to pics
    data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        # lambda x: x.to(device),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        # lambda x: x.to(device),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }
    tensor_list = []
    for img in list_of_img:
        tensor_list.append(data_transforms['test'](img))
        
    list_of_predictions = []
    for tensor in tensor_list:
        pred = pic_model.predict(tensor)
        list_of_predictions.append(pred)
        
    print(list_of_predictions)
    
    # MOUSE ENGINE
    hidden_dim = 16
    mouse_model = Generator(3, hidden_dim, hidden_dim)
    mouse_model.load_state_dict('models/mouse_model.pth')
    
    current_position = gui_agent.get_mouse_position()
    target_position = (300, 300)
    
    model_target_position = current_position - target_position
    # change to tensor
    model_target_position = torch.tensor(model_target_position)

    path = mouse_model(model_target_position)[0, :, 1:3].detach().numpy()
    
    
    
    mouse = MouseEngine()
    # mouse.target = (300, 300)
    # while True:
        # mouse.move_the_mouse_one_step()
        # sleep(0.001)
    

if __name__ == '__main__':
    main()