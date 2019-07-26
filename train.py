import argparse
from utils.controller import train
from config.config_util import choose_config
from models.distancenet import Resnet18
from utils.utils import seed_program

# Use this for a binary classification network. What if I train a network to embedd cats & dogs differently via cosine similarity. Batch size of 16 -> 8 cats & 8 dogs -> 28 positive comparisons for cats & dogs each and 8*8=64 negative dog-cat combinations. Positive combinations scales sorta good, check list(combinations(inp, 2))
# At inference time, the processed image is checked against k reference images/embeddings corresponding to either cat or dog. Hopefully the network learns to cluster images from one class close to one another - and I hope it's somewhat stable as long as there are a large amount of reference embeddings.
# Would scale beyond binary classification if computer is powerful or if we do one class at a time and dont backprop every step (for stability).
# At production, if you missclassify an example you can simply add that example to your reference images/embeddings without retraining.

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab']
  p.add_argument('--config', type=str, default='laptop', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)

  # Create model
  model = Resnet18(config)

  train(model, config)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)