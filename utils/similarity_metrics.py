import torch

def all_active_metrics():
  all_active = [
    MedianMetric(),
    MeanMetric(),
    Top1kMetric(),
    Top3kMetric(),
    Top10kMetric(),
    Bot10kMetric(),
    # Bot30kMetric(),
    # Top10Bot10Metric(),
  ]
  return all_active

class SimilarityMetric:
  def __call__(self, query_emb, db_emb):
    raise NotImplementedError('Similarity metrics need to do... stuff?')

  def __str__(self):
    return type(self).__name__.replace('Metric', '')

class MedianMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    cat_val, dog_val = cat_dist.median(), dog_dist.median()
    return cat_val, dog_val

class MeanMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    cat_val, dog_val = cat_dist.mean(), dog_dist.mean()
    return cat_val, dog_val

class Top1kMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    k = 1
    cat_val, dog_val = cat_dist.topk(k)[0].mean(), dog_dist.topk(k)[0].mean()
    return cat_val, dog_val

class Top3kMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    k = 3
    cat_val, dog_val = cat_dist.topk(k)[0].mean(), dog_dist.topk(k)[0].mean()
    return cat_val, dog_val

class Top10kMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    k = 10
    cat_val, dog_val = cat_dist.topk(k)[0].mean(), dog_dist.topk(k)[0].mean()
    return cat_val, dog_val

class Bot10kMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    k = 10
    cat_val = cat_dist.topk(k, largest=False)[0].mean()
    dog_val = dog_dist.topk(k, largest=False)[0].mean()
    return cat_val, dog_val


class Bot30kMetric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    k = 30
    cat_val = cat_dist.topk(k, largest=False)[0].mean()
    dog_val = dog_dist.topk(k, largest=False)[0].mean()
    return cat_val, dog_val

class Top10Bot10Metric(SimilarityMetric):
  def __call__(self, cat_dist, dog_dist):
    top_k, bot_k = 10, 30
    top_cat = cat_dist.topk(top_k, largest=False)[0].mean()
    bot_cat = cat_dist.topk(top_k)[0].mean()
    cat_val = (top_cat + bot_cat) / 2

    top_dog = dog_dist.topk(bot_k, largest=False)[0].mean()
    bot_dog = dog_dist.topk(bot_k)[0].mean()
    dog_val = (top_dog + bot_dog) / 2

    return cat_val, dog_val






if __name__ == '__main__':
  import math
  import torch
  import torch.nn as nn



  a = torch.ones(1, 3)
  b = torch.ones(1, 3)
  a[0][0] = 0
  print(a)
  metric = nn.CosineSimilarity()

  cc = metric(a, b)
  print(cc)
  qweqw



# ratio * någon viktgrej som blir mindre bakot typ log eller sigmoid
  a = 0.9
  b = 0.5

  r1 = a / (a + b)
  print(r1)

  my_e = lambda num: math.exp(num) ** 15

  r2 = my_e(a) / (my_e(a) + my_e(b))
  print(r2)

