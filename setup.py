from setuptools import setup, find_packages

setup(
    name='vgg_util',
    version='0.1.0',
    description='Utilized VGG model for PyTorch',
    packages=find_packages(exclude=('tests')),
    author='So Uchida',
    author_email='s.aiueo32@gmail.com',
    install_requires=["torch", "torchvision"],
    url='https://github.com/S-aiueo32/vgg_util',
)
