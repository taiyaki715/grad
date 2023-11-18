import torch
import torchinfo

from vit import CustomViT
from head import Head

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = CustomViT()
    self.head = Head()

  def __str__(self):
    return torchinfo.summary(self, input_size=(1, 3, 512, 512))

  def forward(self, original_inputs):
    vit_outputs = self.vit(original_inputs)
    x = self.head(vit_outputs, original_inputs)

    return x
