import torch
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_b_16(image_size=256)
    self.vit.heads = torch.nn.Linear(768, 1024);

  def forward(self, x):
    x = self.vit(x)

    return x
