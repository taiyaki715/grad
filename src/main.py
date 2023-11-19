import torch

from losses.ssim_loss import SSIMLoss
from models.model import Model
from trainer import Trainer

params = {
  'batch_size': 1,
  'num_epochs': 3,
  'lerning_rate': 1e-4,
  'device_type': 'mps',
  'criterion': SSIMLoss,
  'optimizer': torch.optim.Adam,
  'model': Model,
  'train_path': '/Volumes/western_digital_4tb/train',
  'test_path': '/Volumes/western_digital_4tb/val'
}

trainer = Trainer(params)
trainer.run()
