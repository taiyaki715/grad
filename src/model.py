import torch
import torchinfo
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
    self.vit.heads = torch.nn.Linear(in_features=1024, out_features=4096)

    # ViTのヘッドを除いてパラメータを固定
    for name, param in self.vit.named_parameters():
      if 'heads' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

  def forward(self, x):
    x = self.vit(x)

    return x


class Head(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
    self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, x):
    x = torch.reshape(x, (-1, 1, 64, 64))
    x = self.conv1(x)
    x = self.upsample1(x)
    x = self.conv2(x)
    x = self.upsample2(x)
    x = self.conv3(x)
    x = self.upsample3(x)

    return x


class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = CustomViT()
    self.head = Head()

  def __str__(self):
    return torchinfo.summary(self, input_size=(1, 3, 512, 512))

  def forward(self, x):
    x = self.vit(x)
    x = self.head(x)

    return x
