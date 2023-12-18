import torch
import torchinfo
import einops

from models.vit import ViT
from models.head import Head

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = ViT(image_size=224, patch_size=14, dim=196, depth=6, n_heads=7, channels=3, mlp_dim=256)
    self.head = Head()

  def __str__(self):
    return torchinfo.summary(self, col_names=["input_size", "output_size", "num_params"], input_size=(1, 3, 224, 224))

  def forward(self, x):
    x = self.vit(x)
    x = torch.reshape(x, (-1, 256, 14, 14))
    x = self.head(x)

    return x
