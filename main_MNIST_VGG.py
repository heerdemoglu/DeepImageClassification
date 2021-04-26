# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch on MNIST Dataset.
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# *********************************************************************************************************************
import os
import time

import numpy as np
import torchvision.utils
from matplotlib import pyplot as plt
from six.moves import urllib

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # Initialize torch device with GPU priority.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories: -- Subdirectories to be created later:
    data_dir = './datasets/MNIST_data'
    trn_dir = os.path.join(data_dir, 'train')
    tst_dir = os.path.join(data_dir, 'test')

    save_dir = './reporting/MNIST/VGG'
    image_save_dir = './reporting/results/MNIST/VGG'

    # Training Parameters:
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epochs = 2

    # Model Parameters:
    trn_tst_split = 0.8  # % range of training data to testing data
    image_scale = 224  # VGG accepts images of this size. (Resize input to this scale)

    # Load the training and testing datasets: (Apply normalization and build to tensors as well)

    # Added to circumvent Cloudflare protection bug, following from: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Normalization values of dataset from : https://discuss.pytorch.org/t/normalization-in-the-mnist-example/
    trnsfrm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.Grayscale(num_output_channels=3),
                                  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=trn_dir, train=True, download=True, transform=trnsfrm)
    train, val = random_split(train_data, [55000, 5000])
    test = datasets.MNIST(root=tst_dir, train=False, download=True, transform=trnsfrm)

    # Construct the loaders:
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)      # Training Loader
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)          # Validation Loader
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)        # Testing Loader

    # Visualize some of the training and testing samples:
    print("Number of Samples in Training Dataset: ", len(train))
    print("Number of Samples in Validation Dataset: ", len(val))
    print("Number of Samples in Testing Dataset: ", len(test))

    # Create Tensorboard Log: - Holds all the example images, histories and network architectures.
    writer = SummaryWriter(os.path.join(save_dir, 'runs/vgg11'))
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    # Construct the images: - Print as figure:
    # Create the subplot:
    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Training Images")
    for (ax, image, label) in zip(axs.flat, images[0:25], labels[0:25]):
        ax.imshow(image[0].cpu(), cmap='gray')
        ax.set_title('Label: {}'.format(label)).set_fontsize('6')
        ax.axis('off')  # hide axes
    # plt.get_current_fig_manager().window.state('zoomed')  # maximize in windows
    # plt.show()
    # fig.savefig(os.path.join(image_save_dir, 'train_data_samples.png'))
    # plt.close(fig)

    # Log images to tensorboard:
    writer.add_figure('input_images', fig, 0)

    # Model loading & Visualization: (All models are set to pretrained = False for the assignment's requirements.)
    vgg = [models.vgg11(pretrained=False), models.vgg11_bn(pretrained=False),
           models.vgg13(pretrained=False), models.vgg13_bn(pretrained=False),
           models.vgg16(pretrained=False), models.vgg16_bn(pretrained=False),
           models.vgg19(pretrained=False), models.vgg19_bn(pretrained=False)
           ]

    # Visualize one of the models - VGG-16 with Batch Normalization "vgg16_bn":
    # print(vgg[0])

    # Start the training procedure and logging process:

    # History tracking variables:
    train_hist = {'train_loss': [], 'validation_loss': [],
                  'train_acc': [], 'validation_acc': [],
                  'per_epoch_times': [], 'total_time': []
                  }

    # Construct the model, optimizer and loss function, then send model to device (GPU or CPU):
    model = vgg[0]
    model.classifier[6] = nn.Linear(4096, 10)
    model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    writer.add_graph(model, images)

    start_time = time.time()
    print('Start training.')
    for epoch in range(epochs):
        losses = list()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # forward pass
            loss = loss_criterion(outputs, labels)  # compute loss

            # compute accuracy:
            acc = (outputs == labels)

            optimizer.zero_grad()  # clean the gradient

            loss.backward()  # backpropagation
            # params.grad._sum(dLoss/dparams)

            optimizer.step()  # step in opposite direction to gradient
            # with torch.no_grad(): params -= eta*params.grad

            # Training Logging:
            losses.append(loss.item())
            print(f'Iteration {i}, loss: {torch.tensor(losses).mean():.2f}')
        print(f'Epoch {epoch+1}, training loss: {torch.tensor(losses).mean():.2f}')

        vlosses = list()
        for j, (vimages, vlabels) in enumerate(val_loader):
            vimages, vlabels = images.to(device), labels.to(device)
            with torch.no_grad():
                voutputs = model(vimages)  # forward pass
            vloss = loss_criterion(voutputs, vlabels)  # compute loss
            optimizer.zero_grad()  # clean the gradient

            # Training Logging:
            vlosses.append(vloss.item())
        print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(vlosses).mean():.2f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Training Time Elapsed: ', str(elapsed_time))
    writer.close()

    # Todo: Testing procedure and logging to tensorboard
