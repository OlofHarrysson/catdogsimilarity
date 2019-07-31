import argparse
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program
from data.data import ImageDataset
import torch
from utils.validator import *
from utils.similarity_metrics import all_active_metrics
from logger import Logger
import numpy as np
from collections import Counter

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
  metrics = all_active_metrics()
  
  cat_e = cat_embs, ref_cat, ref_dog # Validation
  dog_e = dog_embs, ref_cat, ref_dog # Validation
  im_paths = cat_paths + dog_paths # Validation

  # cat_e = ref_cat, cat_embs, dog_embs
  # dog_e = ref_dog, cat_embs, dog_embs
  # im_paths = cat_ref_paths + dog_ref_paths

  cat_bounds = defaultdict(list) # Data for scatterplot
  cat_bounds = calc_bounds(cat_e, metrics, cat_bounds, is_cat_embs=True)
  dog_bounds = defaultdict(list) # Data for scatterplot
  dog_bounds = calc_bounds(dog_e, metrics, dog_bounds, is_cat_embs=False)
  
  # Log boundrary
  all_wrong_paths = []
  for metric in metrics:
    cat_data = cat_bounds[str(metric)]
    dog_data = dog_bounds[str(metric)]
    data = dict(cat=cat_data, dog=dog_data)
    textlabels, wrong_paths = None, []
    # textlabels, data, wrong_paths = missclassified(cat_data, dog_data, im_paths)

    logger.log_boundrary(data, str(metric), 0, textlabels)
    all_wrong_paths.extend(wrong_paths)

  # Images
  # wrong_paths = Counter(all_wrong_paths)
  # images_to_log = [p for p, v in wrong_paths.items() if v > 7] # Most common
  images_to_log = set(all_wrong_paths)
  # logger.log_images(images_to_log) #

  # Accuracy
  embs = cat_embs, ref_cat, ref_dog
  cat_preds = defaultdict(list)
  cat_preds = calc_preds(embs, metrics, cat_preds, is_cat_embs=True)

  embs = dog_embs, ref_cat, ref_dog
  dog_preds = defaultdict(list)
  dog_preds = calc_preds(embs, metrics, dog_preds, is_cat_embs=False)

  # Log Accuracy
  preds = defaultdict(list)
  for metric in metrics:
    metric_name = str(metric)
    preds[metric_name].extend(cat_preds[metric_name])
    preds[metric_name].extend(dog_preds[metric_name])
  
  logger.log_accuracy(preds, 1)


  # Predict
  for metric in metrics:
    cat_data = cat_bounds[str(metric)]
    dog_data = dog_bounds[str(metric)]
    calc_loss(cat_data)
    qwe


def missclassified(cat_data, dog_data, im_paths):
  ''' Finds the missclassified datapoints and gets their image paths '''
  data = np.array(cat_data + dog_data)
  mis_catdata, mis_dogdata, mis_im_paths = [], [], []
  assert len(im_paths) == data.shape[0], 'recalc embeddings'
  textlabels = []
  for d, im_path in zip(data, im_paths):
    if d[0] < d[1]: # Wrong prediction
      label = im_path.stem.split('.')[1] # im id
      class_ = im_path.stem.split('.')[0] # cat/dog
      textlabels.append(label)
      mis_im_paths.append(im_path)
      if class_ == 'cat':
        mis_catdata.append(d)
      else:
        mis_dogdata.append(d)

  mis_data = dict(cat=mis_catdata, dog=mis_dogdata)
  return textlabels, mis_data, mis_im_paths

def calc_loss(data):
  loss = 0
  for d in data:
    print(d)
    asd

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)