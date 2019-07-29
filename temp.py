import torch
import torch.nn as nn
import visdom
import numpy as np

viz = visdom.Visdom(port='6006')

metric = nn.CosineSimilarity()
sims = []
for i in range(10000):
  var = 512
  # var = 2048
  a = torch.randn(1, var)
  b = torch.randn(1, var)
  sim = metric(a, b)
  sims.append(sim)

viz.boxplot(X=np.array(sims).reshape(1, -1))