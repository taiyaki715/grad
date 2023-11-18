import torch
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
    self.vit.heads = torch.nn.Identity()

    # ViTのヘッドを除いてパラメータを固定
    for _, param in self.vit.named_parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.vit(x)

    return x
