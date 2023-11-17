import torch
import matplotlib.pyplot as plt

from data_loader import Dataset
from model import Model

batch_size = 2
num_epochs = 5
lerning_rate = 0.001
device_type = 'mps'

# データセットの定義
dataset = Dataset(train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Validationデータセットの定義
dataset_val = Dataset(train=False)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

# モデルの定義
model = Model()

# モデルをGPUに転送
device = torch.device(device_type);
model.to(device);

# 損失関数
criterion = torch.nn.MSELoss()

# 最適化アルゴリズム
optimizer = torch.optim.Adam(model.parameters(), lr=lerning_rate)

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

    print(f"Epoch:{current_epoch + 1}/{num_epochs} Batch:{i + 1}/{len(data_loader)} Loss:{running_loss / (i + 1)}")

  inputs, targets = next(iter(data_loader_val))
  inputs, targets = inputs.to(device), targets.to(device)
  outputs = model(inputs)
  inputs, outputs, targets = inputs.cpu().numpy(), outputs.cpu().detach().numpy(), targets.cpu().numpy()

  fig, axes = plt.subplots(3, batch_size, figsize=(16, 6))
  for i in range(batch_size):
    axes[0][i].imshow(inputs[i].transpose(1, 2, 0))
    axes[1][i].imshow(outputs[i][0])
    axes[2][i].imshow(targets[i][0])
    axes[0][i].axis('off')
    axes[1][i].axis('off')
    axes[2][i].axis('off')
    axes[0][0].set_title('input')
    axes[1][0].set_title('output')
    axes[2][0].set_title('target')
  plt.savefig(f"output_{current_epoch + 1}.png")

  losses.append(running_loss / len(data_loader))
  print(f'Epoch: [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')

print(losses)
