import visdom
import torch
import torch.nn as nn
import numpy as np
from utils.utils import EMAverage
import plotly.graph_objects as go

def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()] # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs

class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)

  def log_cosine(self, catpair, dogpair, catdogpair):
    metric = nn.CosineSimilarity()

    cat_sim = metric(catpair[0], catpair[1])
    cat_sim = cat_sim.cpu().detach().numpy()
    dog_sim = metric(dogpair[0], dogpair[1])
    dog_sim = dog_sim.cpu().detach().numpy()
    catdog_sim = metric(catdogpair[0], catdogpair[1])
    catdog_sim = catdog_sim.cpu().detach().numpy()

    title_text = 'Cosine Distance'
    fig = go.Figure()

    violin_plot = lambda ys, side, name: go.Violin(y=ys,
                            box_visible=True,
                            meanline_visible=True,
                            spanmode='hard',
                            x0='Cat',
                            name=name,
                            )

    fig.add_trace(violin_plot(cat_sim, 'negative', 'Cat'))
    fig.add_trace(violin_plot(dog_sim, 'positive', 'Dog'))
    fig.add_trace(violin_plot(catdog_sim, 'positive', 'Catdog'))
    # TODO: Visdom doesn't work with title in layout

    fig.update_layout(
      shapes=[
        # Line Horizontal
        go.layout.Shape(
          type="line",
          x0=-0.5,
          y0=0.75,
          x1=0.5,
          y1=0.75,
          line=dict(
            width=2,
            dash="dot",
          ),
        ),
          
      ]
    )

    self.viz.plotlyplot(fig, win=title_text)