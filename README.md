# vgg_util
## Introduction
Perceptual loss is used for many image generation tasks.
It is task-dependent which layer of VGG is used to compare, so VGG model easy to pick up any layers is helpful.
Here provides the VGG models based on `torchvision`'s pretrained models.

## Requirements
- Python
- `torch` & `torchvision`

## installation
```
pip install git+https://github.com/S-aiueo32/vgg_util
```

## Usage
First, create the model with the types like `vgg11`, `vgg19` etc.
Next, input the image with the layer name you want to pick up.
```python
import torch

from vgg_util import VGG


model = VGG(model_type='vgg19')

x = torch.rand(1, 3, 256, 256)
y = model(x, targets=['relu2_2', 'relu5_4'])  # return dict of tensors

print(y['relu2_2'].shape)  # => (1, 128, 128, 128)
print(y['relu5_4'].shape)  # => (1, 512, 16, 16)
```
