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
    # self.basenet = models.resnet18(pretrained=config.pretrained)
    # self.basenet = models.resnet34(pretrained=config.pretrained)
    # self.basenet = models.resnet50(pretrained=config.pretrained)

    self.basenet = resnet18(pretrained=config.pretrained)
    # self.basenet = resnet50(pretrained=config.pretrained)

    fc_features = self.basenet.fc.in_features
    # self.basenet.fc = nn.Linear(fc_features, fc_features) # For torchvision
    self.cosine = nn.Linear(fc_features, fc_features)
    self.direct = nn.Linear(fc_features, 1)
    
  def forward(self, inputs):
    inputs = inputs.to(self.device)
    x = self.basenet(inputs)
    return self.cosine(x), self.direct(x)

  def predict(self, inputs):
    with torch.no_grad():
      return self.forward(inputs)[0]

  def predict_normal(self, inputs):
    with torch.no_grad():
      return torch.sigmoid(self.forward(inputs)[1])

  def save(self, path):
    print("Saving Weights @ " + path)
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)
