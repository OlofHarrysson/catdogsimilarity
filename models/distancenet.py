import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from .resnet import *
from .resnet import BasicBlock


class Resnet18(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    n_features = config.n_model_features
    self.basenet = models.resnet18(pretrained=config.pretrained)
    # self.basenet = models.resnet34(pretrained=config.pretrained)
    self.basenet.fc = nn.Linear(self.basenet.fc.in_features, n_features)
    
  def forward(self, inputs):
    inputs = inputs.to(self.device)
    return self.basenet(inputs).chunk(2)