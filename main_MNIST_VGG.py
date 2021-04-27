# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch on MNIST Dataset.
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# *********************************************************************************************************************
import os
import time

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
    batch_size = 160
    learning_rate = 0.0001
    momentum = 0.9
    epochs = 50

    # Model Parameters:
    image_scale = 224       # VGG accepts images of this size. (Resize input to this scale)

    # Load the training and testing datasets: (Apply normalization and build to tensors as well)
    # Added to circumvent Cloudflare protection bug, following from: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Normalization values of dataset from : https://discuss.pytorch.org/t/normalization-in-the-mnist-example/
    # ToDo: Additional transforms can be added for data augmentation.
    trnsfrm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.Grayscale(num_output_channels=3),
                                  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=trn_dir, train=True, download=True, transform=trnsfrm)
    train, val = random_split(train_data, [55000, 5000])
    test = datasets.MNIST(root=tst_dir, train=False, download=True, transform=trnsfrm)

    # Construct the loaders:
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)  # Training Loader
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)  # Validation Loader
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)  # Testing Loader

    # Visualize some of the training and testing samples:
    print("Number of Samples in Training Dataset: ", len(train))
    print("Number of Samples in Validation Dataset: ", len(val))
    print("Number of Samples in Testing Dataset: ", len(test))

    # Model loading & Visualization: (All models are set to pretrained = False for the assignment's requirements.)
    vgg = [models.vgg11(pretrained=False), models.vgg11_bn(pretrained=False),
           models.vgg13(pretrained=False), models.vgg13_bn(pretrained=False),
           models.vgg16(pretrained=False), models.vgg16_bn(pretrained=False),
           models.vgg19(pretrained=False), models.vgg19_bn(pretrained=False)
           ]

    vgg_names = [#"vgg11", #"vgg11_bn",
                 #"vgg13", #"vgg13_bn",
                 #"vgg16", #"vgg16_bn",
                 #"vgg19",
                 "vgg19_bn"
                 ]
    # Loop over all possible models listed.
    for i in range(len(vgg)):
        # Training Parameters:
        batch_size = 160
        learning_rate = 0.0001
        momentum = 0.9
        epochs = 50

        # Model Parameters:
        image_scale = 224  # VGG accepts images of this size. (Resize input to this scale)

        # Create Tensorboard Log: - Holds all the example images, histories and network architectures.
        writer = SummaryWriter(os.path.join(save_dir, os.path.join('runs', vgg_names[i])))
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
        # Log images to tensorboard:
        writer.add_figure('input_images', fig, 0)

        # Visualize one of the models - VGG-16 with Batch Normalization "vgg16_bn":
        # print(vgg[0])

        # Start the training procedure and logging process:
        # Construct the model, optimizer and loss function, then send model to device (GPU or CPU):
        model = vgg[i]                                      # pick the model to work on
        model.classifier[6] = nn.Linear(4096, 10)           # set to number of classes in mnist
        model.to(device)                                    # send to cuda
        writer.add_graph(model, images)                     # add the model to tensorboard

        loss_criterion = nn.CrossEntropyLoss()              # create the loss criterion and the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        start_time = time.time()
        print('Start training of: ', vgg_names[i], '.')
        for epoch in range(epochs):
            # ep_train_loss = 0.0
            # ep_train_acc = 0
            train_loss = 0.0
            train_corr = 0
            for k, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)                      # forward pass
                loss = loss_criterion(outputs, labels)       # compute training loss

                optimizer.zero_grad()  # clean the gradient

                loss.backward()  # backpropagation
                # params.grad._sum(dLoss/dparams)

                optimizer.step()  # step in opposite direction to gradient
                # with torch.no_grad(): params -= eta*params.grad

                # Training Logging:
                train_loss += loss.item()   # Accumulate running loss.
                # ep_train_loss += train_loss

                # Calculate and accumulate correct outputs: & Log them in tensorboard:
                _, predicted = torch.max(outputs.data, 1)
                train_corr += (predicted == labels).sum().item()
                if (k+1) % 100 == 0:
                    running_accuracy = train_corr / 100 / outputs.size(0)
                    # ep_train_acc += running_accuracy
                    print(f'Epoch [{epoch + 1}/{epochs}],  Step: [{k + 1}/{len(train_loader)}], '
                          f'Training Loss: {loss.item():.4f}, Training Accuracy: {running_accuracy:.4f}')
                    writer.add_scalar('trn_loss', loss.item(), epoch * len(train_loader) + k+1)
                    writer.add_scalar('trn_acc', running_accuracy, epoch * len(train_loader) + k+1)
                    train_corr = 0
                    train_loss = 0.0
                # writer.add_scalar('ep_trn_loss', ep_train_loss/len(train_loader), epoch + 1)
                # writer.add_scalar('ep_trn_acc', ep_train_acc/len(train_loader), epoch + 1)

            # Todo: Use validation loss to achieve early stopping; LRScheduling etc.
            val_loss = 0.0
            val_corr = 0
            # ep_val_acc = 0
            for j, (vimages, vlabels) in enumerate(val_loader):
                vimages, vlabels = vimages.to(device), vlabels.to(device)
                with torch.no_grad():
                    voutputs = model(vimages)  # forward pass
                vloss = loss_criterion(voutputs, vlabels)  # compute loss
                optimizer.zero_grad()  # clean the gradient

                # Validation Logging:
                val_loss += vloss.item()  # Accumulate running loss.

                # Calculate and accumulate correct outputs: & Log them in tensorboard:
                _, predicted = torch.max(voutputs.data, 1)
                val_corr += (predicted == vlabels).sum().item()
                running_accuracy = val_corr / 100 / voutputs.size(0)
                # ep_val_acc += running_accuracy
            print(f'Validation Loss: {vloss.item():.4f}, Validation Accuracy: {running_accuracy:.4f}')
                # writer.add_scalar('ep_val_loss', val_loss/len(val_loader), epoch+1)
                # writer.add_scalar('ep_val_acc', ep_val_acc/len(val_loader), epoch+1)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Training+Validation Time Elapsed: ', str(elapsed_time))
            writer.add_scalar('epoch_elapsed', elapsed_time, epoch)

        # # Testing Procedure:
        # # Todo logs to tensorboard -- Testing acc, inference speed, example img pairs, conf matrix, model size
        # correct = 0
        # total = 0
        # for j, (vimages, vlabels) in enumerate(test_loader):
        #     vimages, vlabels = vimages.to(device), vlabels.to(device)
        #     with torch.no_grad():
        #         voutputs = model(vimages)  # forward pass
        #     _, predicted = torch.max(voutputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == vlabels).sum().item()
        # print(correct / total * 100)
        #
        writer.close()
# todo: Save best models to drive; write a reader.
# ToDo: Download logs from Google Colaboratory.
