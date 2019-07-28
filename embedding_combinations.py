import torch
from itertools import *

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
  b_size = 32
  batch = range(1, b_size+1)
  batch2 = [b+.5 for b in batch]

  inp1 = torch.tensor(batch).unsqueeze(1)
  inp2 = torch.tensor(batch2).unsqueeze(1)
  pairs = create_pairs(inp1, inp2)

  p1, p2, p3 = pairs
  # print(p1)
  # print(p2)
  print(p3)


def test2():
  b_size = 32 # Per class
  batch = range(1, b_size+1)
  combs = list(combinations(batch, 2))
  # print(combs)

  perms = list(product(batch, repeat=2))
  # print(perms)

  print(f'n={b_size} -> {len(combs)} same-class comparisons (twice) & {len(perms)} between-class comparisons -> {len(combs)*2 + len(perms)} total comparisons')

if __name__ == '__main__':
  # test()
  test2()
