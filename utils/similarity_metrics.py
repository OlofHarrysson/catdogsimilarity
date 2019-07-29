def all_active_metrics():
  all_active = [
    MedianMetric(),
    MeanMetric(),
    Top1kMetric(),
    Top3kMetric(),
    Top10kMetric(),
    Bot10kMetric(),
    Bot30kMetric(),
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











