import torch

from models.model import Model
from trainer import Trainer
3

params = {
  'batch_size': 16,
  'num_epochs': 100,
  'lerning_rate': 1e-4,
  'device_type': 'mps',
  'criterion': torch.nn.MSELoss,
  'optimizer': torch.optim.Adam,
  'model': Model,
}

trainer = Trainer(params)
trainer.run()
