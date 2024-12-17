import os
from dotenv import load_dotenv
import torch
from torchviz import make_dot

from gym.captchas.models.single.modelSingle import ModelSingle
from gym.captchas.models.multi.modelMulti import ModelMulti, ModelMultiSimple, ModelMultiTwoHead
load_dotenv()

class Visualization:
    def __init__(self):
        pass

    def visualizeModelMulti(self) -> None:
        model = ModelMultiSimple(12)
        model.print_model_summary()
        
    def visualizeModelSingle(self) -> None:
        model = ModelSingle(11)
    

if __name__ == '__main__':
        viz = Visualization()
        viz.visualizeModelMulti()
        # viz.visualizeModelSingle()