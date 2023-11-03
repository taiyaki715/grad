import torch
import torchinfo
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    self.vit.heads = torch.nn.Linear(in_features=768, out_features=784)

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
    self.pool1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.pool2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.pool3 = torch.nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, x):
    x = torch.reshape(x, (-1, 1, 28, 28))
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.pool3(x)

    return x


class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = CustomViT()
    self.head = Head()

  def __str__(self):
    return torchinfo.summary(self, input_size=(1, 3, 224, 224))

  def forward(self, x):
    x = self.vit(x)
    x = self.head(x)

    return x
