# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# *********************************************************************************************************************
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Time Counter
import tensorboard as tb

import torch

if __name__ == '__main__':
    # Initialize torch device with GPU priority.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories: -- Subdirectories to be created later:
    data_dir = './datasets/MNIST_data'
    save_dir = './reporting/MNIST/VGG'
    image_save_dir = './reporting/results/MNIST/VGG'

    # Training Parameters:
    batch_size = 100
    learning_rate = 0.0002
    epochs = 100

    # Model Parameters:
    image_size = 28

    # Load the training and testing datasets: (Apply normalization and build to tensors as well)
    # Normalization parameters from: ToDo.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Todo: validation and test loadars
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
