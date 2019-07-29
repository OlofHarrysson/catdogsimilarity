import argparse
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program
from data.data import setup_valdata
import torch
from utils.validator import calc_embeddings

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab', 'predict']
  p.add_argument('--config', type=str, default='predict', choices=configs, help='What config file to choose')

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

  ref_dataloader, dataloader = setup_valdata(config)
  ref_cat, ref_dog = calc_embeddings(model, ref_dataloader)
  cat_embs, dog_embs = calc_embeddings(model, dataloader)
  
  torch.save(ref_cat, 'output/ref_cat.pth')
  torch.save(ref_dog, 'output/ref_dog.pth')
  torch.save(cat_embs, 'output/cat_embs.pth')
  torch.save(dog_embs, 'output/dog_embs.pth')

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)