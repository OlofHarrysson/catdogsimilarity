import torch
import torch.nn as nn
from data.data import setup_valdata
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import dataclasses
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from utils.similarity_metrics import all_active_metrics

class Validator():
  def __init__(self, model, logger, config):
    self.model, self.logger, self.config = model, logger, config
    self.ref_dataloader, self.dataloader = setup_valdata(config)
    self.best_acc = 0

  def validate(self, step):
    self.model.eval()
    ref_cat, ref_dog = calc_embeddings(self.model, self.ref_dataloader)
    val_cats, val_dogs = calc_embeddings(self.model, self.dataloader)
    
    # Boundrary
    metrics = all_active_metrics()
    embs = val_cats, ref_cat, ref_dog
    cat_bounds = defaultdict(list) # Data for scatterplot
    cat_bounds = calc_bounds(embs, metrics, cat_bounds, is_cat_embs=True)
    
    embs = val_dogs, ref_cat, ref_dog
    dog_bounds = defaultdict(list) # Data for scatterplot
    dog_bounds = calc_bounds(embs, metrics, dog_bounds, is_cat_embs=False)
    
    # Log boundrary
    for metric in metrics:
      cat_data = cat_bounds[str(metric)]
      dog_data = dog_bounds[str(metric)]
      data = dict(cat=cat_data, dog=dog_data)
      self.logger.log_boundrary(data, str(metric), step)

    # Accuracy
    embs = val_cats, ref_cat, ref_dog
    cat_preds = defaultdict(list)
    cat_preds = calc_preds(embs, metrics, cat_preds, is_cat_embs=True)

    embs = val_dogs, ref_cat, ref_dog
    dog_preds = defaultdict(list)
    dog_preds = calc_preds(embs, metrics, dog_preds, is_cat_embs=False)

    # Log Accuracy
    preds = defaultdict(list)
    for metric in metrics:
      metric_name = str(metric)
      preds[metric_name].extend(cat_preds[metric_name])
      preds[metric_name].extend(dog_preds[metric_name])
    
    best_acc, best_acc_name = self.logger.log_accuracy(preds, step)

    # Cat or not
    # self.logger.cat_or_not(ref_cat, val_cats, val_dogs, step)

    # Save model on val improvement
    if best_acc > self.best_acc:
      self.save_model(best_acc, best_acc_name, step)

    self.model.train()

  def val_normal(self, step):
    self.model.eval()

    preds = defaultdict(list)
    for data in self.dataloader:
      cat_input, dog_input = data
      cat_preds = self.model.predict_normal(cat_input).cpu()
      dog_preds = self.model.predict_normal(dog_input).cpu()

      for cat_pred in cat_preds:
        if cat_pred < 0.5:
          preds['normal'].append(1)
        else:
          preds['normal'].append(0)

      for dog_pred in dog_preds:
        if dog_pred > 0.5:
          preds['normal'].append(1)
        else:
          preds['normal'].append(0)

    self.logger.log_normal_accuracy(preds, step)
    self.model.train()


  def save_model(self, acc, name, step):
    self.best_acc = acc
    path = f'output/{name}-acc{acc:.4f}-step{step}.pth'
    self.model.save(path)

def calc_embeddings(model, dataloader):
  cat_embs, dog_embs = torch.tensor([]), torch.tensor([])
  for ind, data in enumerate(dataloader, 1):
    if ind % 100 == 0:
      print(ind)

    # if ind > 10:
      # break

    cat_input, dog_input = data

    cat_embs = torch.cat((cat_embs, model.predict(cat_input).cpu()))
    dog_embs = torch.cat((dog_embs, model.predict(dog_input).cpu()))

  return cat_embs, dog_embs

def calc_distance(emb, ref_emb):
  emb = emb.expand_as(ref_emb)
  metric = nn.CosineSimilarity()
  return metric(emb, ref_emb)


def calc_bounds(embs, metrics, boundrary_data, is_cat_embs):
  calc_embs, ref_cat, ref_dog = embs
  for emb in calc_embs:
    cat_dist = calc_distance(emb, ref_cat)
    dog_dist = calc_distance(emb, ref_dog)

    for metric in metrics:
      cat_val, dog_val = metric(cat_dist, dog_dist)
      if is_cat_embs:
        boundrary_data[str(metric)].append([cat_val, dog_val])
      else:
        boundrary_data[str(metric)].append([dog_val, cat_val])

  return boundrary_data 

def calc_preds(embs, metrics, preds, is_cat_embs):
  is_correct = lambda d1, d2: 1 if d1 > d2 else 0
  calc_embs, ref_cat, ref_dog = embs
  for emb in calc_embs:
    cat_dist = calc_distance(emb, ref_cat)
    dog_dist = calc_distance(emb, ref_dog)

    for metric in metrics:
      cat_val, dog_val = metric(cat_dist, dog_dist)
      if is_cat_embs:
        preds[str(metric)].append(is_correct(cat_val, dog_val))
      else:
        preds[str(metric)].append(is_correct(dog_val, cat_val))

  return preds 