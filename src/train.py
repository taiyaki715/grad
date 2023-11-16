import torch
import matplotlib.pyplot as plt

from data_loader import Dataset
from model import Model

# データセットの定義
dataset = Dataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# モデルの定義
model = Model()

# モデルをGPUに転送
device = torch.device('mps');
model.to(device);

# 損失関数
criterion = torch.nn.MSELoss()

# 最適化アルゴリズム
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 学習するエポック数
num_epochs = 10

losses = []

for current_epoch, epoch in enumerate(range(num_epochs)):
  running_loss = 0.0

  for i, batch in enumerate(data_loader):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    print(f"Epoch:{current_epoch + 1}/{num_epochs} Batch:{i}/{len(data_loader)} Loss:{running_loss / (i + 1)}")

  losses.append(running_loss / len(data_loader))
  print(f'Epoch: [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')

print(losses)