import argparse
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program
from data.data import setup_testdata
import torch
from utils.validator import *
from logger import Logger
import numpy as np

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab', 'predict', 'submit']
  p.add_argument('--config', type=str, default='submit', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)

  # Create model
  model = Resnet18(config)
  model.load(config.model_path)
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU
  model.eval()

  dataloader = setup_testdata(config)
  im_embs = pred_embeddings(model, dataloader)
  
  ref_cat = torch.load('output/ref_cat.pth')
  ref_dog = torch.load('output/ref_dog.pth')

  logger = Logger(config)
  preds = []
  for im_emb in im_embs:
    cat_dist = calc_distance(im_emb, ref_cat)
    dog_dist = calc_distance(im_emb, ref_dog)
    
    cat_mean = cat_dist.mean()
    dog_mean = dog_dist.mean()

    in_min, in_max = 0.33, 0.67
    out_min, out_max = -3, 3

    span = lambda max_v, min_v: max_v - min_v 
    in_span = span(in_max, in_min)
    out_span = span(out_max, out_min)

    d = [cat_mean.item(), dog_mean.item()]
    d = np.clip(d, in_min, in_max)

    cat = span(d[0], in_min) * out_span / in_span + out_min
    dog = span(d[1], in_min) * out_span / in_span + out_min
    pred = softmax([cat, dog])

    if pred[0] > pred[1]: # Cat
      preds.append(1 - pred[0])
    else:
      preds.append(pred[1])

    # if cat_mean > dog_mean:
    #   if cat_mean > 0.8:
    #     preds.append(0.001)
    #   else:
    #     preds.append(0.01)
    # else:
    #   preds.append(0.99)

    save_file(preds)

def save_file(preds):
  subm_data = []
  subm_data.append('id,label')
  for ind, pred in enumerate(preds, 1):
    row = f'{ind},{pred}'
    subm_data.append(row)

  with open('submission.csv', 'w') as f:
    f.writelines("\n".join(subm_data))


def pred_embeddings(model, dataloader):
  embs = torch.tensor([])
  for ind, data in enumerate(dataloader, 1):
    # if ind > 3:
    #   break
    print(ind)
    # embs = torch.cat((embs, model.predict(data).cpu()))
    embs = torch.cat((embs, model.predict_normal(data).cpu()))

  return embs

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def secondmain(config):
  config = choose_config(config_str)
  seed_program(config.seed)

  # Create model
  model = Resnet18(config)
  model.load(config.model_path)
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU
  model.eval()

  dataloader = setup_testdata(config)
  im_embs = pred_embeddings(model, dataloader)

  preds = np.array(im_embs).flatten().tolist()
  save_file(preds)


if __name__ == '__main__':
  config_str = parse_args()
  # main(config_str)
  secondmain(config_str)