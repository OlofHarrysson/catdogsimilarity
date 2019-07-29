import argparse
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program
from data.data import ImageDataset
import torch
from utils.validator import *
from utils.similarity_metrics import all_active_metrics
from logger import Logger

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab', 'predict']
  p.add_argument('--config', type=str, default='predict', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)

  cat_ref_paths = ImageDataset(f'{config.reference_dataset}/cats').image_files
  dog_ref_paths = ImageDataset(f'{config.reference_dataset}/dogs').image_files
  cat_paths = ImageDataset(f'{config.val_dataset}/cats').image_files
  dog_paths = ImageDataset(f'{config.val_dataset}/dogs').image_files

  # Create model
  model = Resnet18(config)
  model.load(config.model_path)
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU
  model.eval()
  
  ref_cat = torch.load('output/ref_cat.pth')
  ref_dog = torch.load('output/ref_dog.pth')
  cat_embs = torch.load('output/cat_embs.pth')
  dog_embs = torch.load('output/dog_embs.pth')


  logger = Logger(config)
  # validator = Validator(model, logger, config)
  # ks = [1, 2, 5, 10]
  # validator.inner_val(ref_cat, ref_dog, cat_embs, dog_embs, 0, ks)
  
  metrics = all_active_metrics()
  embs = cat_embs, ref_cat, ref_dog
  cat_bounds = defaultdict(list) # Data for scatterplot
  cat_bounds = calc_bounds(embs, metrics, cat_bounds, is_cat_embs=True)
  
  embs = dog_embs, ref_cat, ref_dog
  dog_bounds = defaultdict(list) # Data for scatterplot
  dog_bounds = calc_bounds(embs, metrics, dog_bounds, is_cat_embs=False)
  
  # Log boundrary
  for metric in metrics:
    cat_data = cat_bounds[str(metric)]
    dog_data = dog_bounds[str(metric)]
    data = dict(cat=cat_data, dog=dog_data)
    logger.log_boundrary(data, str(metric), 0)


if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)