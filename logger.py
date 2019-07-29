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

    self.over_line_average = None
    self.loss_percent_average = None

  def init_over_line_average(self, names):
    self.over_line_average = {}
    for key in names:
      self.over_line_average[key] = EMAverage(50)

  def init_loss_percent_average(self, loss_dict):
    self.loss_percent_average = {}
    for key in loss_dict.keys():
      self.loss_percent_average[key] = EMAverage(50)

  def log_loss(self, loss, step):
    Y = torch.Tensor([loss]).numpy()
    self.viz.line(
      Y=Y.reshape((1,1)),
      X=[step],
      update='append',
      win='TotalLoss',
      opts=dict(
          xlabel='Steps',
          ylabel='Loss',
          title='Training Loss',
          legend=['Total']

      )
    )

  def log_loss_percent(self, loss_dict, step):
    if self.loss_percent_average == None:
      self.init_loss_percent_average(loss_dict)

    legend, losses = [], []
    for name, loss in loss_dict.items():
      legend.append(name)
      avg_tracker = self.loss_percent_average[name]
      val = avg_tracker.update(loss.item())
      losses.append(val)

    tot_loss = sum(losses)
    temp_loss = 0
    Y = []
    for loss in losses:
      Y.append((temp_loss + loss) / tot_loss)
      temp_loss += loss

    self.viz.line(
      Y=np.array(Y).reshape(1, -1),
      X=[step],
      update='append',
      win='losspercent',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Percentage',
          title='Loss Percentage',
          stackgroup='one',
          legend=legend
      )
    )

  def log_cosine(self, catpair, dogpair, catdogpair, step):
    metric = nn.CosineSimilarity()

    cat_sim = metric(catpair[0], catpair[1])
    cat_sim = cat_sim.cpu().detach().numpy()
    dog_sim = metric(dogpair[0], dogpair[1])
    dog_sim = dog_sim.cpu().detach().numpy()
    catdog_sim = metric(catdogpair[0], catdogpair[1])
    catdog_sim = catdog_sim.cpu().detach().numpy()

    title_text = 'Cosine Distance'
    fig = go.Figure()

    violin_plot = lambda ys, name: go.Violin(y=ys,
                            box_visible=True,
                            meanline_visible=True,
                            spanmode='hard',
                            name=name,
                            )

    fig.add_trace(violin_plot(cat_sim, 'Cat'))
    fig.add_trace(violin_plot(dog_sim, 'Dog'))
    fig.add_trace(violin_plot(catdog_sim, 'Catdog'))
    # TODO: Visdom doesn't work with title in layout

    fig.update_layout(
      shapes=[
        # Line Horizontal
        go.layout.Shape(
          type="line",
          x0=-0.5,
          y0=0.5,
          x1=2.5,
          y1=0.5,
          line=dict(
            width=2,
            dash="dot",
          ),
        ),
          
      ]
    )

    self.viz.plotlyplot(fig, win=title_text)
    self.log_over_theline(cat_sim, dog_sim, catdog_sim, step)

  def log_over_theline(self, cat_sim, dog_sim, catdog_sim, step):
    names = ['cat', 'dog', 'catdog']
    if self.over_line_average == None:
      self.init_over_line_average(names)

    line = 0.5
    n_cat = np.sum(cat_sim < line) / len(cat_sim)
    n_dog = np.sum(dog_sim < line) / len(dog_sim)
    n_catdog = np.sum(catdog_sim > line) / len(catdog_sim)
    n_overs = [n_cat, n_dog, n_catdog]

    Y = []
    for name, n_over in zip(names, n_overs):
      avg_tracker = self.over_line_average[name]
      val = avg_tracker.update(n_over)
      Y.append(val)

    self.viz.line(
      Y=np.array(Y).reshape((1, -1)),
      X=[step],
      update='append',
      win='Over the line',
      opts=dict(
          xlabel='Steps',
          ylabel='Percentage over',
          title=f'Over the line',
          ytickmin = 0,
          ytickmax = 0.2,
          legend=['Cat', 'Dog', 'Catdog'],
      )
    )

  def log_val_cosine(self, distances, step):

    title_text = 'Cosine Distance Validation'
    fig = go.Figure()

    violin_plot = lambda ys, name: go.Violin(y=ys,
                            box_visible=True,
                            meanline_visible=True,
                            spanmode='hard',
                            name=name,
                            )

    for name, distance in distances.items():
      fig.add_trace(violin_plot(distance, name))

    # TODO: Visdom doesn't work with title in layout
    fig.update_layout(
      shapes=[
        # Line Horizontal
        go.layout.Shape(
          type="line",
          x0=-0.5,
          y0=0.5,
          x1=2.5,
          y1=0.5,
          line=dict(
            width=2,
            dash="dot",
          ),
        ),
          
      ]
    )

    self.viz.plotlyplot(fig, win=title_text)


  def log_accuracy(self, preds_dict, step):
    accuracy, legends = [], []
    for metric, preds in preds_dict.items():
      accuracy.append(sum(preds) / len(preds))
      legends.append(metric)

    Y = np.array(accuracy).reshape((1, -1))
    self.viz.line(
      Y=Y,
      X=[step],
      update='append',
      win='Accuracy',
      opts=dict(
        xlabel='Steps',
        ylabel='Accuracy',
        title=f'Val Accuracy',
        ytickmin = 0.7,
        ytickmax = 1,
        legend=legends,
      )
    )

    best_acc = max(accuracy)
    name = legends[np.argmax(accuracy)]
    return best_acc, name


  def log_boundrary(self, preds_dict, name, step):
    X, Y = np.empty((0, 2)), np.empty((0, 1))
    for metric, preds in preds_dict.items():
      preds = np.array(preds)
      X = np.append(X, preds, axis=0)

      labels = np.ones((preds.shape[0], 1))
      if metric == 'cat':
        Y = np.append(Y, labels, axis=0)
      else:
        Y = np.append(Y, labels*2, axis=0)

    min_vals = X.min(0)
    max_vals = X.max(0)
    self.viz.scatter(
      X=X,
      Y=Y,
      win=f'Boundrary {name}',
      opts=dict(
        title=f'Val Boundrary {name}',
        legend=['Cat', 'Dog'],
        markersize=3,
        xlabel='Same',
        ylabel='Different',
        layoutopts=dict(
          plotly=dict(
            shapes=[dict(
              line=dict(
                dash='dot',
                width='2'),
              type='line',
              x0= min_vals[0],
              x1= max_vals[1],
              y0= min_vals[0],
              y1= max_vals[1])]
            ))
      )
    )