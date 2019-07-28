import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCosineLoss(nn.Module):
  def __init__(self, margin=0):
    super().__init__()
    self.margin = margin
    self.metric = nn.CosineSimilarity()


  def forward(self, x1, x2):
    sim = self.metric(x1, x2)
    loss = 1 - sim - self.margin
    loss = torch.clamp(loss, 0, 2) # TODO: Check max value
    # TODO: Do nonzero() before mean?
    return loss.mean()


if __name__ == '__main__':
  loss_fn = MyCosineLoss(margin=0.5)
  x1 = torch.randn(4, 3)
  x2 = torch.randn(4, 3)
  
  loss = loss_fn(x1, x2)
  print(loss)