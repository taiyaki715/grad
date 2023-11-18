import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from data_loader import Dataset
from models.model import Model

from losses.ssim_loss import SSIMLoss

# パラメータ
batch_size = 1
num_epochs = 5
lerning_rate = 1e-3
device_type = 'mps'

# データセットの定義
dataset = Dataset(train=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Validationデータセットの定義
dataset_val = Dataset(train=False)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=True)

# モデルの定義
model = Model()

# モデルをGPUに転送
device = torch.device(device_type);
model.to(device);

# 損失関数
criterion = SSIMLoss()

# 最適化アルゴリズム
optimizer = torch.optim.Adam(model.parameters(), lr=lerning_rate)

# 学習
for current_epoch, epoch in enumerate(range(num_epochs)):
  model.train()
  train_loss = 0.0
  with tqdm(data_loader, total=len(data_loader), desc=f'Epoch {current_epoch + 1}/{num_epochs}', unit='batch') as pbar:
    for i, batch in enumerate(pbar):
      pbar.set_postfix_str(f'Loss: {round(train_loss / (i + 1), 5)}')
      inputs, targets = batch
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

  # テストデータで損失の計算
  model.eval()
  val_loss = 0.0
  with tqdm(data_loader_val, total=len(data_loader_val), desc=f'Val {current_epoch + 1}/{num_epochs}', unit='batch') as pbar:
    for i, batch in enumerate(pbar):
      pbar.set_postfix_str(f'Loss: {round(val_loss / (i + 1), 5)}')
      inputs, targets = batch
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      val_loss += loss.item()

  # テストデータをinput8枚分可視化。inputs, outputs, targetsの順に並べ、ラベルをつける。軸はなし。データのスケールはinputひとつごとに調整。最後にファイルに保存。
  fig, axes = plt.subplots(3, 8, figsize=(16, 6))
  inputs, targets = next(iter(data_loader_val))
  inputs, targets = inputs.to(device), targets.to(device)
  outputs = model(inputs)
  inputs = inputs.cpu().numpy()
  targets = targets.cpu().numpy()
  outputs = outputs.cpu().detach().numpy()

  for i in range(8):
    axes[0][i].imshow(inputs[i].transpose(1, 2, 0))
    axes[1][i].imshow(outputs[i][0])
    axes[2][i].imshow(targets[i][0])
    axes[0][i].axis('off')
    axes[1][i].axis('off')
    axes[2][i].axis('off')

  axes[0][0].set_title('Input')
  axes[1][0].set_title('Output')
  axes[2][0].set_title('Target')

  #カラーバーの表示
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(axes[0][0].imshow(inputs[0].transpose(1, 2, 0)), cax=cbar_ax)

  plt.savefig(f'epoch_{epoch + 1}.png')

  print(f'Epoch {epoch + 1}/{num_epochs} MSE: {val_loss / len(data_loader_val)}')

# モデルの保存
torch.save(model.state_dict(), 'model.pth')
