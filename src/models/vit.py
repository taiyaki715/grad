import torch
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    self.vit.heads = torch.nn.Linear(768, 196);
  
    #self.vit.heads以外のパラメータを固定
    for param in self.vit.parameters():
      param.requires_grad = False
    for param in self.vit.heads.parameters():
      param.requires_grad = True

  def forward(self, x):
    x = self.vit(x)

    return x
