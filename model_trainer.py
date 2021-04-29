# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch on MNIST Dataset.
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# **********************************************************************************************************************
import os
import urllib

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision.transforms import transforms
from matplotlib import pyplot as plt


#  ToDo: Method documentation to be implemented:
def load_dataset(dataset_name, image_scale, trn_direc, tst_direc, batch_size, output_channels):
    print('Loading the dataset.')
    # Load the training and testing datasets: (Apply normalization and build to tensors as well)
    # Added to circumvent Cloudflare protection bug, following from: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Implement CIFAR100 Data loading procedure:
    if dataset_name == 'mnist':
        # Normalization values of dataset from : https://discuss.pytorch.org/t/normalization-in-the-mnist-example/
        # Additional transforms can be added for data augmentation; Such as rotation:
        trnsfrm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(image_scale),
                                      transforms.RandomRotation(15),
                                      transforms.Grayscale(num_output_channels=output_channels),
                                      transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=trn_direc, train=True, download=True, transform=trnsfrm)
        train, val = random_split(train_data, [55000, 5000])
        test = datasets.MNIST(root=tst_direc, train=False, download=True, transform=trnsfrm)

        # Construct the loaders:
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)  # Training Loader
        val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)  # Validation Loader
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)  # Testing Loader

        # # Visualize some of the training and testing samples:
        # print("Number of Samples in Training Dataset: ", len(train))
        # print("Number of Samples in Validation Dataset: ", len(val))
        # print("Number of Samples in Testing Dataset: ", len(test))

        return train_loader, val_loader, test_loader
    # ToDo: Implement CIFAR100 Data loading procedure:
    if dataset_name == 'cifar100':
        pass
    else:
        raise ValueError('Dataset must be mnist or cifar100')


#  ToDo: Method documentation to be implemented:
def load_model_template(dataset_name, num_of_classes, cuda_device=None):
    print('Loading model template.')
    if dataset_name == 'vgg11':
        model = models.vgg11(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes in mnist

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if dataset_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes in mnist

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if dataset_name == 'vgg19':
        model = models.vgg11(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes in mnist

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if dataset_name == 'vgg19_bn':
        model = models.vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes in mnist

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    # ToDo: Implement ResNet models:
    # ToDo: Implement GoogLeNet models:

    else:
        raise ValueError('The model name provided is not supported in this assignment.')


#  ToDo: Method documentation to be implemented:
def show_images(data_loader):
    print('Saved input image samples to tensorboard.')
    # Get the images and respective labels:
    images, labels = next(iter(data_loader))

    # Construct the images: - Print as figure:
    # Create the subplot:
    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Training Images")
    for (ax, image, label) in zip(axs.flat, images[0:25], labels[0:25]):
        ax.imshow(image[0], cmap='gray')
        ax.set_title('Label: {}'.format(label)).set_fontsize('6')  # set title for each subplot
        ax.axis('off')  # hide axes
    return fig


# ToDo: Method documentation to be implemented:
# ToDo: Implement the method.
def show_output_images(model, data_loader):
    pass


#  ToDo: Method documentation to be implemented:
def train_model(model, data_loader, epochs, criterion, optimizer, sum_writer, validation_loader,
                model_save_path, cuda_device=None):
    print('Starting training procedure.')
    # Iterate over given number of epochs:
    for epoch in range(epochs):
        model.train()
        # Iterate over all batches of the training dataset:
        ep_loss_run = 0.0
        epoch_correct = 0
        total_samples = 0
        for idx, (images, labels) in enumerate(data_loader):
            # Send images and labels to GPU:
            if cuda_device is not None:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear the gradient.

            # Do a forward pass on the batch:
            outputs = model(images)  # forward pass

            # Make the predictions on the batch:
            _, preds = torch.max(outputs, 1)                     # Pull the predictions.
            model_loss = criterion(outputs, labels)              # Compute training loss.

            model_loss.backward()                                # Execute backpropagation.
            optimizer.step()                                     # Step in opposite direction to gradient.

            ep_loss_run += model_loss.item() * outputs.shape[0]  # Accumulate the epoch loss (avg with batch size).
            epoch_correct += (preds == labels).sum().item()      # Accumulate number of correct class. each batch.
            total_samples += outputs.size(0)                     # Count number of samples seen so far.

            # Training Statistics & Logging: ***********************************************************************
            # At each batch record the data (both in terminal and tensorboard):
            if (idx + 1) % outputs.shape[0] == 0:
                print(f'Epoch [{epoch + 1} / {epochs}], '
                      f'Step: [{idx + 1}/{len(data_loader)}], '
                      f'Training Loss: {model_loss.item():.4f}, '
                      f'Training Accuracy: {(epoch_correct / total_samples):.4f}'
                      )
                sum_writer.add_scalar('trn_loss', ep_loss_run / total_samples,
                                      epoch * outputs.shape[0] + idx + 1)
                sum_writer.add_scalar('trn_acc', (epoch_correct / total_samples),
                                      epoch * outputs.shape[0] + idx + 1)
            # ******************************************************************************************************
        # Report everything at the end of each epoch: **************************************************************
        epoch_loss = ep_loss_run / total_samples
        epoch_acc = epoch_correct / total_samples

        # Write end of epoch results & Log this to tensorboard:
        print(f'Epoch [{epoch + 1}/{epochs}], Epoch Training Loss: {epoch_loss:.4f},'
              f'Epoch Training Accuracy: {epoch_acc:.4f}')
        sum_writer.add_scalar('epoch_trn_loss', epoch_loss, epoch + 1)
        sum_writer.add_scalar('epoch_trn_acc', epoch_acc, epoch + 1)
        # **********************************************************************************************************

        # Try validation set at the end of each epoch:
        validate_model(model, validation_loader, epochs, criterion, optimizer, sum_writer, model_save_path, cuda_device)


# ToDo: Method documentation to be implemented:
# ToDo: Add LRScheduler, Early Stopping, Log which epoch was the best.
def validate_model(model, data_loader, epochs, criterion, optimizer, sum_writer, model_save_path, cuda_device=None):
    print('Starting validation procedure.')
    min_valid_loss = np.inf
    # Iterate over given number of epochs:
    for epoch in range(epochs):
        model.eval()
        # Iterate over all batches of the training dataset:
        ep_loss_run = 0.0
        epoch_correct = 0
        total_samples = 0
        for idx, (images, labels) in enumerate(data_loader):
            # Send images and labels to GPU:
            if cuda_device is not None:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear the gradient.

            # Do a forward pass on the batch:
            outputs = model(images)  # forward pass

            # Make the predictions on the batch:
            _, preds = torch.max(outputs, 1)                     # Pull the predictions.
            model_loss = criterion(outputs, labels)              # Compute training loss.

            ep_loss_run += model_loss.item() * outputs.shape[0]  # Accumulate the epoch loss (avg with batch size).
            epoch_correct += (preds == labels).sum().item()      # Accumulate number of correct class. each batch.
            total_samples += outputs.size(0)                     # Count number of samples seen so far.

            # Training Statistics & Logging: ***********************************************************************
            # At each batch record the data (both in terminal and tensorboard):
            if (idx + 1) % outputs.shape[0] == 0:
                print(f'Epoch [{epoch + 1} / {epochs}], '
                      f'Step: [{idx + 1}/{len(data_loader)}], '
                      f'Validation Loss: {model_loss.item():.4f}, '
                      f'Validation Accuracy: {(epoch_correct / total_samples):.4f}'
                      )
                sum_writer.add_scalar('val_loss', ep_loss_run / total_samples,
                                      epoch * outputs.shape[0] + idx + 1)
                sum_writer.add_scalar('val_acc', (epoch_correct / total_samples),
                                      epoch * outputs.shape[0] + idx + 1)
            # ******************************************************************************************************
        # Report everything at the end of each epoch: **************************************************************
        epoch_loss = ep_loss_run / total_samples
        epoch_acc = epoch_correct / total_samples

        # Save best models to drive:
        if min_valid_loss > epoch_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{epoch_loss:.6f}) \t Saving the model.')
            min_valid_loss = epoch_loss

            # Saving State Dict:
            torch.save(model.state_dict(), model_save_path)

        # Write end of epoch results & Log this to tensorboard:
        print(f'Epoch [{epoch + 1}/{epochs}], Epoch Validation Loss: {epoch_loss:.4f},'
              f'Epoch Validation Accuracy: {epoch_acc:.4f}')
        sum_writer.add_scalar('epoch_val_loss', epoch_loss, epoch + 1)
        sum_writer.add_scalar('epoch_val_acc', epoch_acc, epoch + 1)
        # **********************************************************************************************************


# ToDo: Method documentation to be implemented:
def test_model(model, test_loader, criterion, optimizer, sum_writer, cuda_device=None):
    print('Starting testing procedure.')
    # Assume that the best model is already loaded.
    model.eval()
    # Iterate over all batches of the training dataset:
    ep_loss_run = 0.0
    epoch_correct = 0
    total_samples = 0

    for idx, (images, labels) in enumerate(test_loader):
        # Send images and labels to GPU:
        if cuda_device is not None:
            images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear the gradient.

        # Do a forward pass on the batch:
        outputs = model(images)  # forward pass

        # Make the predictions on the batch:
        _, preds = torch.max(outputs, 1)  # Pull the predictions.
        model_loss = criterion(outputs, labels)  # Compute training loss.

        ep_loss_run += model_loss.item() * outputs.shape[0]  # Accumulate the epoch loss (avg with batch size).
        epoch_correct += (preds == labels).sum().item()  # Accumulate number of correct class. each batch.
        total_samples += outputs.size(0)  # Count number of samples seen so far.

    # Report everything after passing through all the batches:
    epoch_loss = ep_loss_run / total_samples
    epoch_acc = epoch_correct / total_samples

    # Write end of epoch results & Log this to tensorboard:
    print(f'Test Dataset Loss: {epoch_loss:.4f},'
          f'Test Dataset Accuracy: {epoch_acc:.4f}')
    sum_writer.add_text('test_loss', f'Test Dataset Loss: {epoch_loss:.4f}', 0)
    sum_writer.add_text('test_accuracy', f'Test Dataset Accuracy: {epoch_acc:.4f}', 0)
    # **********************************************************************************************************


if __name__ == '__main__':
    # Model Variables: (All used dataset and model samples listed here.)
    DATASET = ['mnist', 'cifar100']
    MODELS = ['vgg11', 'vgg11_bn', 'vgg_19', 'vgg19_bn', 'resnet18', 'resnet152', 'googlenet']

    # Model training parameters: ***************************************************************************************
    EPOCHS = 2
    BATCH_SIZE = 100
    LR = 0.0001
    MOMENTUM = 0.9
    # ******************************************************************************************************************

    # Select the dataset and the model to train:
    selected_dataset = DATASET[0]
    selected_model = MODELS[0]

    # Directory Variables: (Created using the selected parameters above)
    data_dir = os.path.join('datasets', selected_dataset)
    trn_dir = os.path.join(data_dir, 'train')
    tst_dir = os.path.join(data_dir, 'test')
    save_dir = os.path.join('.\\reporting', selected_model, selected_dataset)
    model_save_dir = os.path.join(save_dir, selected_model + '.pth')

    # Create tensorboard writer:
    writer = SummaryWriter(os.path.join(save_dir, os.path.join('runs', selected_model)))

    # Create the device which the training will take place:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the dataset:
    tr_loader, vl_loader, tst_loader = load_dataset(selected_dataset, 224, trn_dir, tst_dir, BATCH_SIZE, 3)
    demo_loader, _, _ = load_dataset(selected_dataset, 224, trn_dir, tst_dir, 16, 3)

    # Show some example input image-label pairs:
    input_fig = show_images(demo_loader)
    writer.add_figure('input_images', input_fig, 0)
    del demo_loader  # Remove memory

    # Create the model:
    my_model = load_model_template(selected_model, 10, device)

    ims, _ = next(iter(tr_loader))  # Provide temporary images.
    writer.add_graph(my_model, ims.to(device))      # Add the model to tensorboard.
    del ims

    # Model hyperparameters:
    loss_criterion = nn.CrossEntropyLoss()  # create the loss criterion and the optimizer
    model_optm = torch.optim.SGD(my_model.parameters(), lr=LR, momentum=MOMENTUM)

    # Train the model:
    train_model(my_model, tr_loader, EPOCHS, loss_criterion, model_optm, writer, vl_loader,
                model_save_dir, device)

    # Test the model:
    test_model(my_model, tst_loader, loss_criterion, model_optm, writer, device)

    # # ToDo: Show some example output/prediction image-label pairs:

    # # ToDo: Confusion Matrix:

    writer.close()
