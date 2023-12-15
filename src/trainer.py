import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data_loader import Dataset

class Trainer:
  def __init__(self, params):
    self.params = params

    # モデル
    self.model = self.params['model']()
  
    # loss
    self.criterion = self.params['criterion']()
  
    # optimizer
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

    # lossの初期化
    self.train_loss = 0.0
    self.test_loss = 0.0

    # epochの初期化
    self.current_epoch = 0

    # TensorBoardの初期化
    self.summary_writer = SummaryWriter(log_dir='logs')

  def run(self):
    for epoch in range(self.params['num_epochs']):
      self.current_epoch = epoch
      self.train_loss = 0.0
      with tqdm(self.data_loader_train) as pbar:
        for i, batch in enumerate(pbar):
          self._train(batch)
          test_batch = self.data_loader_test.__iter__().__next__()
          self._test(test_batch)
          self._tensorboard(i)

      # 可視化
      self._visualize()

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

  def _visualize(self):
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
    plt.savefig(f'epoch_{self.current_epoch + 1}.png')

  def _tensorboard(self, i):
    self.summary_writer.add_scalar('Loss/train', self.train_loss / (i + 1), self.current_epoch * len(self.data_loader_train) + i)
    self.summary_writer.add_scalar('Loss/test', self.test_loss / (i + 1), self.current_epoch * len(self.data_loader_train) + i)
