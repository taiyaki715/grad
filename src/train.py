import torch
import matplotlib.pyplot as plt

from data_loader import Dataset
from model import Model

# データセットの定義
dataset = Dataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# モデルの定義
model = Model()

# 損失関数
criterion = torch.nn.MSELoss()

# 最適化アルゴリズム
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 学習するエポック数
num_epochs = 10

for epoch in range(num_epochs):
  running_loss = 0.0

  for batch in data_loader:
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    with torch.no_grad():
      outputs = model(inputs)
      plt.figure(figsize=(16, 16))
      plt.subplot(1, 3, 1)
      plt.title('Input')
      plt.imshow(inputs[0].permute(1, 2, 0).numpy() * 255.0)
      plt.subplot(1, 3, 2)
      plt.title('Output')
      plt.imshow(outputs[0].numpy().squeeze() * 255.0, cmap='gray')
      plt.subplot(1, 3, 3)
      plt.title('Target')
      plt.imshow(targets[0].numpy().squeeze() * 255.0, cmap='gray')

      plt.savefig(f'result.png')
      plt.close()

  print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')