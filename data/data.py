import json, random, torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path
import torchvision.transforms as transforms
import imgaug as ia
import imgaug.augmenters as iaa


def setup_traindata(config):
  def collate(batch):
    im_sizes = config.image_input_size
    im_size = random.randint(im_sizes[0], im_sizes[1])
    uniform_size = transforms.Compose([
      transforms.Resize((im_size, im_size)),
      transforms.ToTensor()
    ])

    class1_ims = [uniform_size(b[0]) for b in batch]
    class2_ims = [uniform_size(b[1]) for b in batch]
    return torch.stack(class1_ims), torch.stack(class1_ims)

  batch_size = config.batch_size
  dataset = BinaryDataset(config.dataset)
  sampler = BinarySampler(dataset)
  return DataLoader(dataset, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate, sampler=sampler)
 

class BinarySampler(Sampler):
  def __init__(self, data_source):
    self.data_source = data_source

  def __iter__(self):
    self.data_source.shuffle_ims()
    n = len(self.data_source)
    indexes = list(range(n))
    random.Random().shuffle(indexes)
    return iter(indexes)

  def __len__(self):
    return len(self.data_source)

class BinaryDataset(Dataset):
  def __init__(self, im_root):
    self.cats_dataset = ImageDataset(f'{im_root}/cats')
    self.dogs_dataset = ImageDataset(f'{im_root}/dogs')

    assert_msg = 'Datasets need to be the same length for now'
    assert len(self.cats_dataset) == len(self.dogs_dataset), assert_msg

  def __len__(self):
    return len(self.cats_dataset)

  def __getitem__(self, index):
    cat = self.cats_dataset[index]
    dog = self.dogs_dataset[index]

    return cat, dog

  def shuffle_ims(self):
    random.shuffle(self.cats_dataset.image_files)
    random.shuffle(self.dogs_dataset.image_files)

class ImageDataset(Dataset):
  def __init__(self, im_dirs):
    self.image_files = []

    im_types = ['.jpg', '.png']
    is_image = lambda path: path.suffix in im_types

    if type(im_dirs) == str:
      im_dirs = [im_dirs]

    for im_dir in im_dirs:
      assert Path(im_dir).exists(), "Directory doesn't exist"
      image_files = [f for f in Path(im_dir).glob('**/*') if is_image(f)]
      self.image_files.extend(image_files)

    assert self.image_files,'{} dataset is empty'.format(im_dir)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    im_path = self.image_files[index]
    if index == 0:
      print(im_path)
    im = Image.open(im_path)

    return im