import torch

class Head(torch.nn.Module):
  def __init__(self):
    super().__init__()

    # 32x32 -> 64x64
    self.up_1_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.up_1_conv_1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    self.up_1_relu = torch.nn.ReLU()

    # 64x64 -> 128x128
    self.up_2_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.up_2_conv_1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.up_2_relu = torch.nn.ReLU()

    # 128x128 -> 256x256
    self.up_3_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.up_3_conv_1 = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    self.up_3_relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.up_1_upsample(x)
    x = self.up_1_conv_1(x)
    x = self.up_1_relu(x)

    x = self.up_2_upsample(x)
    x = self.up_2_conv_1(x)
    x = self.up_2_relu(x)
    
    x = self.up_3_upsample(x)
    x = self.up_3_conv_1(x)
    x = self.up_3_relu(x)

    return x
