from pyautogui import sleep
import torch
from guiAgent.guiAgent import GuiAgent
from imageProcessing.imageProcessor import ImageProcessor
from imageProcessing.ocr import OCR
from models.gan import Generator
from models.model import TunedModel
from mouseEngine.mouseEngine import MouseEngine
from torchvision import models, transforms, datasets




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
    
    ocr = OCR()
    header_label = ocr.ocr_from_image('imageProcessing/captcha_pics/header.png')
    header_label = header_label.lower()
    header_label = header_label if header_label[-1] != 's' else header_label[:-1]

    # IMAGE MODEL 
    pic_model = TunedModel(13)
    pic_model.load_state_dict(torch.load('models/saved_models/captcha_model.pt', map_location=torch.device('cpu')))
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
            
    mouse.move_mouse_all_the_way(gui_agent.locate_on_screen("guiAgent/images/verify.png"))
    
    
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