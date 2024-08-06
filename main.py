from pyautogui import sleep
import torch
from app.guiAgent.guiAgent import GuiAgent
from app.imageProcessing import ImageProcessor
from app.imageProcessing import OCR
from gym.captchas.model import TunedModel
from gym.mouse import MouseEngine
from torchvision import transforms
import os
from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    print("CAPTCHA Solver")
    
    gui_agent = GuiAgent()
    gui_agent.open_browser("localhost")
    sleep(2)
    gui_agent.click_checkbox()
    sleep(2)
    filename = gui_agent.take_screenshot(os.getenv('SCREENSHOTS_FOLDER'))
    sleep(2)
    image_processor = ImageProcessor(filename)
    # image_processor.show_image()
    list_of_img =  image_processor.process_captcha_image(os.getenv('CAPTCHA_PICS_FOLDER'))
    
    ocr = OCR()
    header_label = ocr.ocr_from_image(os.getenv('CAPTCHA_HEADER_IMG'))
    header_label = header_label.lower()
    header_label = header_label if header_label[-1] != 's' else header_label[:-1]

    # IMAGE MODEL 
    pic_model = TunedModel(13)
    pic_model.load_state_dict(torch.load(os.getenv('CAPTCHA_RESULT_MODEL'), map_location=torch.device('cpu')))
    pic_model.eval()

    # apply this to pics
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(150),
        transforms.CenterCrop(150),
        # lambda x: x.to(device),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    label_to_index = {
        'bicycle': 0,
        'bridge': 1,
        'bus': 2,
        'car': 3,
        'chimney': 4,
        'crosswalk': 5,
        'a fire hydrant': 6,
        'motorcycle': 7,
        'mountain': 8,
        'other': 9,
        'palm': 10,
        'stair': 11,
        'traffic light': 12
    }
    
    tensor_list = []
    for img in list_of_img:
        tensor_list.append(data_transform(img))
        
    TEMP = 1.0
    THRESHOLD = 0.3
        
    label_index = label_to_index[header_label]
    print(header_label, label_index)
        
    list_of_predictions = []
    for tensor in tensor_list:
        pred = pic_model(tensor.unsqueeze(0))[0]
        pred = torch.nn.functional.softmax(pred / TEMP).detach().numpy()
        should_select = pred[label_index] > THRESHOLD
        list_of_predictions.append(should_select)
        
    print(list_of_predictions)
    

    
    mouse = MouseEngine()
    for i, img in enumerate(list_of_img):
        if list_of_predictions[i]:
            mouse.move_mouse_all_the_way(gui_agent.locate_on_screen(img))
            sleep(1)
            
    mouse.move_mouse_all_the_way(gui_agent.locate_on_screen(os.getenv('ICONS_FOLDER') + 'submit.png'))
    
    
    # MOUSE ENGINE
    # hidden_dim = 16
    # mouse_model = Generator(3, hidden_dim, hidden_dim)
    # mouse_model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    # mouse_model.eval()
    
    # image_pos = gui_agent.locate_on_screen(list_of_img[0])
    # target_position = torch.Tensor(image_pos)
    
    # current_position = torch.Tensor(gui_agent.get_mouse_position())
    # screen_resolution = gui_agent.get_screen_resolution()
    # model_target_position = current_position - target_position
    # model_target_position[0] /= screen_resolution[0]
    # model_target_position[1] /= screen_resolution[1]
    
    # path = mouse_model(model_target_position)[:, 1:3].detach().numpy()
    # path[:, 0] *= screen_resolution[0] 
    # path[:, 1] *= screen_resolution[1]
    

if __name__ == '__main__':
    main()