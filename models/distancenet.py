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
    self.basenet = models.resnet18(pretrained=config.pretrained)
    # self.basenet = models.resnet34(pretrained=config.pretrained)
    fc_features = self.basenet.fc.in_features
    self.basenet.fc = nn.Linear(fc_features, fc_features)
    self.drop_layer = nn.Dropout(p=0.5)
    self.outlayer = nn.Linear(fc_features, config.n_model_features)
    
  def forward(self, inputs):
    inputs = inputs.to(self.device)
    x = F.leaky_relu(self.basenet(inputs))
    x = self.drop_layer(x)
    return self.outlayer(x)

  def predict(self, inputs):
    with torch.no_grad():
      return self.forward(inputs)
