import torch
import torchinfo
import torchvision

class CustomViT(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.vit = torchvision.models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
    self.vit.heads = torch.nn.Linear(in_features=1024, out_features=4096)

    torch.nn.init.uniform_(self.vit.heads.weight, 0.0, 0.01)
    torch.nn.init.zeros_(self.vit.heads.bias)

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

    # ダウンサンプリング側
    self.downsample_conv = torch.nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
    self.downsample1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.downsample2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.downsample3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # アップサンプリング側
    self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
    self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.dropout1 = torch.nn.Dropout2d(p=0.5)
    self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.dropout2 = torch.nn.Dropout2d(p=0.5)
    self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, encoder_outputs, original_inputs):
    down_x_1 = self.downsample_conv(original_inputs)
    down_x_2 = self.downsample1(down_x_1)
    down_x_3 = self.downsample2(down_x_2)
    down_x_4 = self.downsample3(down_x_3)

    x = torch.reshape(encoder_outputs, (-1, 1, 64, 64)) + down_x_4
    x = self.conv1(x)
    x = self.upsample1(x) + down_x_3
    x = self.dropout1(x)
    x = self.conv2(x)
    x = self.upsample2(x) + down_x_2
    x = self.dropout2(x)
    x = self.conv3(x)
    x = self.upsample3(x) + down_x_1

    return x


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
