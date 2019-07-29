import torch, math
from pathlib import Path
import numpy as np
from utils.utils import ProgressbarWrapper as Progressbar
from data.data import setup_traindata
from logger import Logger
from utils.validator import Validator
import torchvision.transforms as transforms
from embedding_combinations import create_pairs
from utils.loss import MyCosineLoss



def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]


def init_training(model, config):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # TODO CyclicLR
  params = filter(lambda p: p.requires_grad, model.parameters())
  # optimizer = torch.optim.Adam(params, lr=config.start_lr)
  optimizer = torch.optim.SGD(params, lr=config.start_lr, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_lr)

  return optimizer, scheduler


def train(model, config):
  # clear_output_dir()
  optimizer, lr_scheduler = init_training(model, config)
  logger = Logger(config)
  validator = Validator(model, logger, config)
  cos_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.45)
  pos_loss_fn = MyCosineLoss(margin=0.45)

  # Data
  dataloader = setup_traindata(config)
  

  # Init progressbar
  n_batches = len(dataloader)
  n_epochs = math.ceil(config.optim_steps / n_batches)
  pbar = Progressbar(n_epochs, n_batches)

  optim_steps = 0
  val_freq = config.validation_freq

  # Training loop
  for epoch in pbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(dataloader, 1):
      pbar.update(epoch, batch_i)

      # Validation
      if optim_steps % val_freq == 0:
        validator.validate(optim_steps)

      # Decrease learning rate
      if optim_steps % config.lr_step_frequency == 0:
        lr_scheduler.step()

      optimizer.zero_grad()
      cats, dogs = data
      inputs = torch.cat((cats, dogs))

      outputs = model(inputs)
      cat_embs, dog_embs = outputs.chunk(2)

      catpair, dogpair, catdogpair = create_pairs(cat_embs, dog_embs)

      # Cosine similarity loss
      cat_loss = pos_loss_fn(catpair[0], catpair[1])
      dog_loss = pos_loss_fn(dogpair[0], dogpair[1])

      y = torch.ones(catdogpair[0].size(0)).to(model.device)
      cat_dog_loss = cos_loss_fn(catdogpair[0], catdogpair[1], -y)

      loss_dict = dict(cat_loss=cat_loss, dog_loss=dog_loss, cat_dog_loss=cat_dog_loss)
      # loss_dict = dict(cat_loss=cat_loss, dog_loss=dog_loss)
      # loss_dict = dict(cat_dog_loss=cat_dog_loss)

      loss = sum(loss_dict.values())
      loss.backward()

      optimizer.step()
      optim_steps += 1

      logger.log_cosine(catpair, dogpair, catdogpair, optim_steps)
      logger.log_loss(loss, optim_steps)
      logger.log_loss_percent(loss_dict, optim_steps)
      
      # Frees up GPU memory
      del data; del outputs

if __name__ == '__main__':
  train()