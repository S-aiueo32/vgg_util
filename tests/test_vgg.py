import torch

from vgg_util import VGG


def test_on_cpu():
    model_types = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                   'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    x = torch.rand(1, 3, 256, 256)
    for model_type in model_types:
        model = VGG(model_type)
        for name in model.names:
            y = model(x, targets=[name])

            if 'conv' in name:
                name_ = name.replace('conv', '').split('_')[0]
                n_downsample = int(name_) - 1
            elif 'relu' in name:
                name_ = name.replace('relu', '').split('_')[0]
                n_downsample = int(name_) - 1
            elif 'pool' in name:
                name_ = name.replace('pool', '')
                n_downsample = int(name_)

            assert x.shape[-1] // 2 ** n_downsample == y[name].shape[-1]


def test_on_gpu():
    model_types = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                   'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    x = torch.rand(1, 3, 256, 256).to('cuda:0')
    for model_type in model_types:
        model = VGG(model_type).to('cuda:0')
        for name in model.names:
            y = model(x, targets=[name])

            if 'conv' in name:
                name_ = name.replace('conv', '').split('_')[0]
                n_downsample = int(name_) - 1
            elif 'relu' in name:
                name_ = name.replace('relu', '').split('_')[0]
                n_downsample = int(name_) - 1
            elif 'bn' in name:
                name_ = name.replace('bn', '').split('_')[0]
                n_downsample = int(name_) - 1
            elif 'pool' in name:
                name_ = name.replace('pool', '')
                n_downsample = int(name_)

            assert x.shape[-1] // 2 ** n_downsample == y[name].shape[-1]
