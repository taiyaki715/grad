import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from data_loader import Dataset

class Trainer:
  def __init__(self, params):
    self.params = params

    # モデルの定義
    self.model = self.params['model']()
    # 損失関数
    self.criterion = self.params['criterion']()
    # 最適化アルゴリズム
    self.optimizer = self.params['optimizer'](self.model.parameters(), lr=params['lerning_rate'])

    # モデルをGPUに転送
    self.device = torch.device(self.params['device_type']);
    self.model.to(self.device);

    # データセットの定義
    dataset_train = Dataset(self.params['train_path'], self.params['test_path'], train=True)
    self.data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.params['batch_size'], shuffle=True)

    # Validationデータセットの定義
    dataset_test = Dataset(self.params['train_path'], self.params['test_path'], train=False)
    self.data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True)

    self.train_loss = 0.0
    self.test_loss = 0.0

  def run(self):
    for epoch in range(self.params['num_epochs']):
      # 学習
      self.train_loss = 0.0
      with tqdm(self.data_loader_train, total=len(self.data_loader_train), desc=f'Epoch {epoch + 1}/{self.params["num_epochs"]}', unit='batch') as pbar:
        for i, batch in enumerate(pbar):
          pbar.set_postfix_str(f'Loss: {round(self.train_loss / (i + 1), 5)}')
          self._train(batch)

      # テスト
      self.test_loss = 0.0
      with tqdm(self.data_loader_test, total=len(self.data_loader_test), desc=f'Test {epoch + 1}/{self.params["num_epochs"]}', unit='batch') as pbar:
        for i, batch in enumerate(pbar):
          pbar.set_postfix_str(f'Loss: {round(self.test_loss / (i + 1), 5)}')
          self._test(batch)

      # 可視化
      self._visualize(epoch)
    
    # モデルの保存
    torch.save(self.model.state_dict(), 'model.pth')

  def _train(self, batch):
    self.model.train()

    # バッチデータをデバイスに転送
    inputs, targets = batch
    inputs, targets = inputs.to(self.device), targets.to(self.device)

    # 学習
    self.optimizer.zero_grad()
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)
    loss.backward()
    self.optimizer.step()
    self.train_loss += loss.item()

  def _test(self, batch):
    self.model.eval()

    inputs, targets = batch
    inputs, targets = inputs.to(self.params['device_type']), targets.to(self.params['device_type'])
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)
    self.test_loss += loss.item()

  def _visualize(self, epoch):
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    inputs, targets = next(iter(self.data_loader_test))
    inputs, targets = inputs.to(self.params['device_type']), targets.to(self.params['device_type'])
    outputs = self.model(inputs)
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    outputs = outputs.cpu().detach().numpy()

    # 画像の表示
    for i in range(8):
      axes[0][i].imshow(inputs[i].transpose(1, 2, 0))
      axes[1][i].imshow(outputs[i][0])
      axes[2][i].imshow(targets[i][0])
      axes[0][i].axis('off')
      axes[1][i].axis('off')
      axes[2][i].axis('off')

    # タイトルの表示
    axes[0][0].set_title('Input')
    axes[1][0].set_title('Output')
    axes[2][0].set_title('Target')

    #カラーバーの表示
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axes[0][0].imshow(inputs[0].transpose(1, 2, 0)), cax=cbar_ax)

    # 画像の保存
    plt.savefig(f'epoch_{epoch + 1}.png')
