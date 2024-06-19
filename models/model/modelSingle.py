import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
from torchvision import models, transforms, datasets
from tqdm import tqdm
from matplotlib import pyplot as plt
import torchsummary
from models.model.modelTools import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ModelSingle(nn.Module, ModelTools):
    def __init__(self):
        pass
