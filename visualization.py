import os
from dotenv import load_dotenv
import torch
from torchviz import make_dot

from gym.captchas.models.single.modelSingle import ModelSingle
from gym.captchas.models.multi.modelMulti import ModelMulti
load_dotenv()

class Visualization:
    def __init__(self):
        pass

    def visualizeModelMulti(self) -> None:
        model = ModelMulti(12)
        model_path = os.getenv('CAPTCHA_MODEL_MULTI')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        make_dot(model(torch.rand(1, 3, 120, 120), torch.rand(1, 12)), params=dict(model.named_parameters())).render("modelMulti", format="png")
        
    def visualizeModelSingle(self) -> None:
        model_path = os.getenv('CAPTCHA_MODEL_SINGLE')
        model = ModelSingle(12)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        make_dot(model(torch.rand(1, 3, 120, 120), torch.rand(1, 12)), params=dict(model.named_parameters())).render("modelSingle", format="png")
    

if __name__ == '__main__':
        viz = Visualization()
        viz.visualizeModelMulti()
        viz.visualizeModelSingle()