import torch

class Head(torch.nn.Module):
  def __init__(self):
    super().__init__()

    # 14x14 -> 28x28
    self.conv_1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
    self.batch_norm_1 = torch.nn.BatchNorm2d(128)
    self.relu_1 = torch.nn.ReLU()

    # 28x28 -> 56x56
    self.conv_2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
    self.batch_norm_2 = torch.nn.BatchNorm2d(64)
    self.relu_2 = torch.nn.ReLU()

    # 56x56 -> 112x112
    self.conv_3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
    self.batch_norm_3 = torch.nn.BatchNorm2d(32)
    self.relu_3 = torch.nn.ReLU()

    # 112x112 -> 224x224
    self.conv_4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    self.tanh = torch.nn.Tanh()

  def forward(self, x):
    x = self.conv_1(x)
    x = self.batch_norm_1(x)
    x = self.relu_1(x)

    x = self.conv_2(x)
    x = self.batch_norm_2(x)
    x = self.relu_2(x)
    
    x = self.conv_3(x)
    x = self.batch_norm_3(x)
    x = self.relu_3(x)

    x = self.conv_4(x)
    x = self.tanh(x)

    return x
