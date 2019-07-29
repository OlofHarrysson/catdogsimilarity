import argparse
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program
from data.data import setup_testdata
import torch
from utils.validator import *
from logger import Logger

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

    if cat_mean > dog_mean:
      preds.append(0.01)
    else:
      preds.append(0.99)

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
    print(ind)
    # if ind > 2:
    #   break
    embs = torch.cat((embs, model.predict(data).cpu()))

  return embs

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)