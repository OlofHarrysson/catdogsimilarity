import torch
import torch.nn as nn
from data.data import setup_valdata
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import dataclasses
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

class Validator():
  def __init__(self, model, logger, config):
    self.model, self.logger, self.config = model, logger, config
    self.ref_dataloader, self.dataloader = setup_valdata(config)
    self.best_acc = 0

  def validate(self, step):
    self.model.eval()
    ref_cat, ref_dog = self.calc_embeddings(self.ref_dataloader)
    val_cats, val_dogs = self.calc_embeddings(self.dataloader)

    dsts = defaultdict(lambda: torch.tensor([]))
    preds = defaultdict(list)
    median_bound = defaultdict(list)
    topk_bound_list = defaultdict(lambda: defaultdict(list))

    for cat_emb, dog_emb in zip(val_cats, val_dogs):
      cat2cat_dist = self.calc_distance(cat_emb, ref_cat)
      cat2dog_dist = self.calc_distance(cat_emb, ref_dog)

      dog2dog_dist = self.calc_distance(dog_emb, ref_dog)
      dog2cat_dist = self.calc_distance(dog_emb, ref_cat)

      # Right prediction if first argument is more similar
      is_correct = lambda d1, d2: 1 if d1 > d2 else 0
      
      # Median pred
      cat_median_pred = is_correct(cat2cat_dist.median(), cat2dog_dist.median())
      dog_median_pred = is_correct(dog2dog_dist.median(), dog2cat_dist.median())
      
      # Topk pred
      topk = lambda t, k: t.topk(k)[0].mean()
      for k in [10, 20, 50, 80]:
        top = defaultdict(list)
        cat_topk_pred = is_correct(topk(cat2cat_dist, k), topk(cat2dog_dist, k))
        dog_topk_pred = is_correct(topk(dog2dog_dist, k), topk(dog2cat_dist, k))

        preds[f'topk-{k}'].append(cat_topk_pred)
        preds[f'topk-{k}'].append(dog_topk_pred)

        topk_bound_list[k]['cat'].append([topk(cat2cat_dist, k), topk(cat2dog_dist, k)])
        topk_bound_list[k]['dog'].append([topk(dog2dog_dist, k), topk(dog2cat_dist, k)])

      # Accuracy
      preds['median'].append(cat_median_pred)
      preds['median'].append(dog_median_pred)

      # Wrong predictions into X-Y scatter plot for clustering
      median_bound['cat'].append([cat2cat_dist.median(), cat2dog_dist.median()])
      median_bound['dog'].append([dog2dog_dist.median(), dog2cat_dist.median()])

      # Violin plot
      dsts['cat2cat'] = torch.cat((dsts['cat2cat'], cat2cat_dist))
      dsts['cat2dog'] = torch.cat((dsts['cat2dog'], cat2dog_dist))
      
      dsts['dog2dog'] = torch.cat((dsts['dog2dog'], dog2dog_dist))
      dsts['dog2cat'] = torch.cat((dsts['dog2cat'], dog2cat_dist))

    # self.logger.log_val_cosine(dsts, step)
    best_acc, best_acc_name = self.logger.log_accuracy(preds, step)
    self.logger.log_boundrary(median_bound, 'median', step)
    for k, data in topk_bound_list.items():
      self.logger.log_boundrary(data, f'topk-{k}', step)

    # Save model on val improvement
    if best_acc > self.best_acc:
      self.save_model(best_acc, best_acc_name)

    self.model.train()

  def calc_embeddings(self, dataloader):
    cat_embs, dog_embs = torch.tensor([]), torch.tensor([])
    for ind, data in enumerate(dataloader, 1):
      cat_input, dog_input = data

      cat_embs = torch.cat((cat_embs, self.model.predict(cat_input).cpu()))
      dog_embs = torch.cat((dog_embs, self.model.predict(dog_input).cpu()))

    return cat_embs, dog_embs


  def calc_distance(self, emb, ref_emb):
    emb = emb.expand_as(ref_emb)
    metric = nn.CosineSimilarity()
    return metric(emb, ref_emb)

    mean, median = distance.mean(), distance.median()
    return mean, median    


  def save_model(self, acc, name):
    self.best_acc = acc
    path = f'output/{name}-acc{acc}.pth'
    print("Saving Weights @ " + path)
    torch.save(self.model.state_dict(), path)
