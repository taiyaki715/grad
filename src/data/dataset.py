from PIL import Image
import torch
import torchvision

from datasets import load_dataset

class Dataset(torch.utils.data.Dataset):
  def __init__(self, is_train=True):
    self.dataset = load_dataset('sayakpaul/nyu_depth_v2')
    self.is_train = is_train

  def __len__(self):
    if self.is_train:
      return len(self.dataset['train'])
    else:
      return len(self.dataset['validation'])

  def __getitem__(self, idx):
    # 画像と深度マップを取得
    if self.is_train:
      image = self.dataset['train'][idx]['image']
      depth = self.dataset['train'][idx]['depth_map']
    else:
      image = self.dataset['validation'][idx]['image']
      depth = self.dataset['validation'][idx]['depth_map']

    # 画像と深度マップを前処理
    image, depth = self._transform(image, depth)

    return image, depth

  def _transform(self, image, depth):
    image = torchvision.transforms.functional.resize(image, (224, 224), interpolation=Image.BILINEAR)
    depth = torchvision.transforms.functional.resize(depth, (224, 224), interpolation=Image.BILINEAR)

    image_tensor = torchvision.transforms.functional.to_tensor(image)
    depth_tensor = torchvision.transforms.functional.to_tensor(depth) / 10.0

    return image_tensor, depth_tensor
