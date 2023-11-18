import torch

class Head(torch.nn.Module):
  def __init__(self):
    super().__init__()

    # ダウンサンプリングブロック
    self.down_1_conv_1 = torch.nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
    self.down_1_conv_2 = torch.nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
    self.down_1_relu = torch.nn.ReLU()

    self.down_2_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_2_conv_1 = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    self.down_2_conv_2 = torch.nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    self.down_2_relu = torch.nn.ReLU()

    self.down_3_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_3_conv_1 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
    self.down_3_conv_2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    self.down_3_relu = torch.nn.ReLU()
    
    self.down_4_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_4_conv_1 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.down_4_conv_2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.down_4_relu = torch.nn.ReLU()
    
    self.down_5_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_5_conv_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.down_5_conv_2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.down_5_relu = torch.nn.ReLU()

    # アップサンプリングブロック
    self.up_1_conv_1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    self.up_1_conv_2 = torch.nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
    self.up_1_relu = torch.nn.ReLU()

    self.up_2_conv_1 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    self.up_2_conv_2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.up_2_relu = torch.nn.ReLU()
    self.up_2_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    self.up_3_conv_1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.up_3_conv_2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
    self.up_3_relu = torch.nn.ReLU()
    self.up_3_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    self.up_4_conv_1 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
    self.up_4_conv_2 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
    self.up_4_relu = torch.nn.ReLU()
    self.up_4_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    self.up_5_conv_1 = torch.nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
    self.up_5_conv_2 = torch.nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
    self.up_5_relu = torch.nn.ReLU()
    self.up_5_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, encoder_outputs, original_inputs):
    # ダウンサンプリングブロック
    down_x = self.down_1_conv_1(original_inputs)
    down_x = self.down_1_conv_2(down_x)
    down_x1 = self.down_1_relu(down_x)

    down_x = self.down_2_maxpool(down_x1)
    down_x = self.down_2_conv_1(down_x)
    down_x = self.down_2_conv_2(down_x)
    down_x2 = self.down_2_relu(down_x)

    down_x = self.down_3_maxpool(down_x2)
    down_x = self.down_3_conv_1(down_x)
    down_x = self.down_3_conv_2(down_x)
    down_x3 = self.down_3_relu(down_x)

    down_x = self.down_4_maxpool(down_x3)
    down_x = self.down_4_conv_1(down_x)
    down_x = self.down_4_conv_2(down_x)
    down_x4 = self.down_4_relu(down_x)

    down_x = self.down_5_maxpool(down_x4)
    down_x = self.down_5_conv_1(down_x)
    down_x = self.down_5_conv_2(down_x)
    down_x5 = self.down_5_relu(down_x)

    # アップサンプリングブロック
    x = torch.reshape(encoder_outputs, (-1, 1, 32, 32))
    
    x = self.up_1_conv_1(x)
    x = self.up_1_conv_2(x)
    x = self.up_1_relu(x) + down_x5

    x = self.up_2_conv_1(x)
    x = self.up_2_conv_2(x)
    x = self.up_2_relu(x)
    x = self.up_2_upsample(x) + down_x4

    x = self.up_3_conv_1(x)
    x = self.up_3_conv_2(x)
    x = self.up_3_relu(x)
    x = self.up_3_upsample(x) + down_x3

    x = self.up_4_conv_1(x)
    x = self.up_4_conv_2(x)
    x = self.up_4_relu(x)
    x = self.up_4_upsample(x) + down_x2

    x = self.up_5_conv_1(x)
    x = self.up_5_conv_2(x)
    x = self.up_5_relu(x)
    x = self.up_5_upsample(x) + down_x1

    return x
