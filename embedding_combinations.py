import torch
from itertools import combinations

def create_positive_pair(embeddings):
  # TODO: Everything is duplicated twice. Need to mask out better?
  batch_size = embeddings.size(0)
  n_repeat = batch_size - 1

  anchors = embeddings.repeat_interleave(n_repeat, dim=0)
  negatives = embeddings.repeat(n_repeat + 1, 1)

  mask = [i for i in range(batch_size**2) if i%(batch_size+1) != 0]
  negatives = negatives[mask]
  return anchors, negatives

def create_negative_pair(embs1, embs2):
  batch_size = embs1.size(0)
  n_repeat = batch_size

  anchors = embs1.repeat_interleave(n_repeat, dim=0)
  negatives = embs2.repeat(n_repeat, 1)

  return anchors, negatives

def create_pairs(embs1, embs2):
  pair1 = create_positive_pair(embs1)
  pair2 = create_positive_pair(embs2)
  negative_pair = create_negative_pair(embs1, embs2)
  return pair1, pair2, negative_pair



def test():
  b_size = 3
  batch = range(1, b_size+1)
  batch2 = [b+.5 for b in batch]

  pairs = create_pairs(torch.tensor(batch), torch.tensor(batch2))

  print(pairs.shape)

  # print("a: {},  p: {},  n: {}".format(a, p, n))
  # print('n={} -> {} comparisons'.format(len_b, len(n)))

def test2():
  b_size = 3
  batch = range(1, b_size+1)
  combs = list(combinations(batch, 2))
  print(combs)

  inds = range(b_size)
  ind_comb = list(combinations(inds, 2))
  print(ind_comb)

if __name__ == '__main__':
  test()
  # test2()
