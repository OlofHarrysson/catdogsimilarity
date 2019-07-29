import json, random, torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_transforms
import imgaug as ia
import imgaug.augmenters as iaa


def setup_traindata(config):
  def collate(batch):
    im_sizes = config.image_input_size
    im_size = random.randint(im_sizes[0], im_sizes[1])
    uniform_size = transforms.Compose([
      transforms.RandomHorizontalFlip(0.5),
      transforms.Resize((im_size, im_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    class1_ims = [uniform_size(b[0]) for b in batch]
    class2_ims = [uniform_size(b[1]) for b in batch]
    return torch.stack(class1_ims), torch.stack(class2_ims)

  batch_size = config.batch_size
  dataset = BinaryDataset(config.dataset)
  sampler = BinarySampler(dataset)
  return DataLoader(dataset, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate, sampler=sampler, drop_last=True)


def setup_valdata(config):
  def collate(batch):
    im_size = config.image_validation_size
    uniform_size = transforms.Compose([
      transforms.Resize((im_size, im_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    class1_ims = [uniform_size(b[0]) for b in batch]
    class2_ims = [uniform_size(b[1]) for b in batch]
    return torch.stack(class1_ims), torch.stack(class2_ims)

  ref_dataset = BinaryDataset(config.reference_dataset)
  batch_size = config.batch_size
  ref_loader = DataLoader(ref_dataset, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate)

  dataset = BinaryDataset(config.val_dataset)
  loader = DataLoader(dataset, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate)

  return ref_loader, loader


def setup_testdata(config):
  def collate(batch):
    uniform_size = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ims = [uniform_size(b) for b in batch]
    return torch.stack(ims)

  dataset = ImageDatasetSorted(config.val_dataset)
  loader = DataLoader(dataset, batch_size=1, num_workers=config.num_workers, collate_fn=collate)

  return loader


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

  # TODO: Check so that the shuffle creates dynamic pairs over epochs
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
    get_file_nbr = lambda path: int(path.stem.split('.')[1])
    self.image_files.sort(key=get_file_nbr)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    im_path = self.image_files[index]
    im = Image.open(im_path)

    return im


class ImageDatasetSorted(Dataset):
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

    get_file_nbr = lambda path: int(path.stem)
    self.image_files.sort(key=get_file_nbr)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    im_path = self.image_files[index]
    im = Image.open(im_path)

    return im