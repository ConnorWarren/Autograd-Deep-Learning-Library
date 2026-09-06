# Deep Learning Library with Automatic Differentiation
This project is a small implementation of a reverse mode automatic differentiation system with support for creating custom neural networks.
It is mainly intended as a learning tool to understand the foundations of libraries like PyTorch.
A full blog post on this project can be found on my [personal site](https://connorwarren.dev).

# Usage
This package is not available on PIP. To use, clone with git:
```
git clone https://github.com/ConnorWarren/Autograd-Deep-Learning-Library/
```
To run the MNIST program in main.py, you must download the training images and labels from [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset),
and place them in the same directory as main.py. Then, you can run main.py as such:

```
usage: main.py [-h] [--train] [--load] [--test] [--ui] [--plot] [--epochs EPOCHS] path
```
