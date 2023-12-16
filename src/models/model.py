import torch
import torchinfo

from models.vit import CustomViT
from models.head import Head

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = CustomViT()
    self.head = Head()

  def __str__(self):
    return torchinfo.summary(self, input_size=(1, 3, 224, 224))

  def forward(self, x):
    x = self.vit(x)
    x = torch.reshape(x, (-1, 1, 14, 14))
    x = self.head(x)

    return x
