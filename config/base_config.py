import pyjokes, random
from datetime import datetime as dtime
from collections import OrderedDict

class DefaultConfig():
  def __init__(self, config_str):
    # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
    # An optional comment to differentiate this run from others
    self.save_comment = pyjokes.get_joke()
    print('\n{}\n'.format(self.save_comment))

    # Start time to keep track of when the experiment was run
    self.start_time = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Use GPU. Set to False to only use CPU
    self.use_gpu = True

    self.max_val_batches = 999

    self.num_workers = 0

    # The config name
    self.config = config_str 

    self.dataset = 'data/datasets/catsdogs/train'
    self.reference_dataset = 'data/datasets/catsdogs/reference'
    self.val_dataset = 'data/datasets/catsdogs/val'

    self.n_model_features = 10

    # Range of Data input size image sizes
    self.image_input_size = (224, 350)
    self.image_validation_size = 300

    self.batch_size = 4

    self.pretrained = True

    self.start_lr = 1e-3
    self.end_lr = 1e-4

    self.optim_steps = 10000
    self.lr_step_frequency = 1

    self.validation_freq = 100

    self.n_model_features = 512

    # Seed to create reproducable training results
    self.seed = random.randint(0, 2**32 - 1)


  def get_parameters(self):
    return OrderedDict(sorted(vars(self).items()))

  def __str__(self):
    # TODO: class name, something else?
    return str(vars(self))

class Laptop(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    ''' Change default parameters here. Like this
    self.seed = 666          ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.use_gpu = False
    self.image_input_size = (30, 35)
    # self.n_model_features = 1
    self.batch_size = 4
    # self.batch_size = 16
    self.val_dataset = 'data/datasets/catsdogs/val_mini'

    self.image_validation_size = 30
    self.validation_freq = 20

class Colab(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.num_workers = 16
    self.batch_size = 32
    self.pretrained = False

    self.start_lr = 5e-2
    self.end_lr = 5e-3
    self.validation_freq = 100


class Predict(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.use_gpu = False
    self.batch_size = 32
    # self.val_dataset = 'data/datasets/catsdogs/val_mini'
    self.reference_dataset = 'data/datasets/catsdogs/train'
    # self.reference_dataset = 'data/datasets/catsdogs/reference'
    # self.model_path = 'output/Mean-acc0.9900-step700.pth'
    self.model_path = 'saved_models/Mean-acc0.9900-step700.pth'



class Submission(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.use_gpu = False
    self.batch_size = 32
    self.val_dataset = 'data/datasets/catsdogs/test1'
    # self.model_path = 'saved_models/Mean-acc0.9900-step700.pth'
    self.model_path = 'saved_models/acc0.9925-step2600-only-directloss.pth'
    
