import torch

class Head(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_0 = torch.nn.Conv2d(1, 128, kernel_size=3, padding_mode='replicate', stride=1, padding=1)
    self.relu_0 = torch.nn.ReLU()

    # 14x14 -> 28x28
    self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv_1 = torch.nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', stride=1, padding=1)
    self.relu_1 = torch.nn.ReLU()

    # 28x28 -> 56x56
    self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv_2 = torch.nn.Conv2d(64, 32, kernel_size=3, padding_mode='replicate', stride=1, padding=1)
    self.relu_2 = torch.nn.ReLU()

    # 56x56 -> 112x112
    self.upsample_3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv_3 = torch.nn.Conv2d(32, 16, kernel_size=3, padding_mode='replicate', stride=1, padding=1)
    self.relu_3 = torch.nn.ReLU()

    self.upsample_4 = torch.nn.Upsample(scale_factor=2, mode='nearest')
    self.conv_4 = torch.nn.Conv2d(16, 1, kernel_size=3, padding_mode='replicate', stride=1, padding=1)
    self.relu_4 = torch.nn.ReLU()

  def forward(self, x):
    x = self.conv_0(x)
    x = self.relu_0(x)

    x = self.upsample_1(x)
    x = self.conv_1(x)
    x = self.relu_1(x)

    x = self.upsample_2(x)
    x = self.conv_2(x)
    x = self.relu_2(x)
    
    x = self.upsample_3(x)
    x = self.conv_3(x)
    x = self.relu_3(x)

    x = self.upsample_4(x)
    x = self.conv_4(x)
    x = self.relu_4(x)

    return x
