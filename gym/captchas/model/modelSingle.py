import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torchsummary
from gym.captchas.model import ModelTools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ModelSingle(nn.Module, ModelTools):
    def __init__(self):
        pass
