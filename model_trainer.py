# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch on MNIST Dataset.
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# **********************************************************************************************************************
import os
import time
import urllib

from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


def load_dataset(model_name, image_scale, trn_direc, tst_direc, batch_size, output_channels):
    print('Loading the dataset.')
    # Load the training and testing datasets: (Apply normalization and build to tensors as well)
    # Added to circumvent Cloudflare protection bug, following from: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    if model_name == 'mnist':
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

        return train_loader, val_loader, test_loader, train, val, test
    if model_name == 'cifar10':
        # Normalization values of dataset from : https://github.com/kuangliu/pytorch-cifar/issues/19
        # Additional transforms can be added for data augmentation; Such as rotation:
        trnsfrm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(image_scale),
                                      transforms.RandomRotation(15), transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_data = datasets.CIFAR10(root=trn_direc, train=True, download=True, transform=trnsfrm)
        train, val = random_split(train_data, [45000, 5000])
        test = datasets.CIFAR10(root=tst_direc, train=False, download=True, transform=trnsfrm)

        # Construct the loaders:
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)  # Training Loader
        val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)  # Validation Loader
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)  # Testing Loader
        return train_loader, val_loader, test_loader, train, val, test
    else:
        raise ValueError('Dataset must be mnist or cifar10')


def load_model_template(model_name, num_of_classes, cuda_device=None):
    print('Loading model template.')
    if model_name == 'vgg11':
        model = models.vgg11(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model
    if model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(512, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model

    if model_name == 'resnet152':
        model = models.resnet152(pretrained=False)
        model.fc = nn.Linear(512, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model

    if model_name == 'googlenet':
        model = models.googlenet(pretrained=False)
        model.fc = nn.Linear(1024, num_of_classes)  # set to number of classes

        # Send to Cuda if GPU is used.
        if cuda_device is not None:
            model.to(cuda_device)
        return model

    # ToDo: Implement your own model.
    if model_name == 'emrenet':
        raise NotImplementedError()
    else:
        raise ValueError('The model name provided is not supported in this assignment.')


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


def show_model_examples(image, corr_idx, false_idx, act_label, pred_label):
    print('Saved output image samples to tensorboard.')

    # Construct the images: - Print as figure:
    # Create the subplot:
    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Correctly and Incorrectly Classified Images")

    i = 0
    for ax in axs.flat:
        if i <= 7:
            ax.imshow(image[corr_idx[i]], cmap='gray')
            ax.set_title(f'Actual: {act_label[corr_idx[i]]}, Predicted: {pred_label[corr_idx[i]]}', c='g').set_fontsize(
                '6')  # set title for each subplot
            i += 1
        else:
            ax.imshow(image[false_idx[i]], cmap='gray')
            title = f'Actual: {act_label[false_idx[i]]}, Predicted: {pred_label[false_idx[i]]}'
            ax.set_title(title, c='r').set_fontsize('6')  # set title for each subplot
            i += 1
    return fig


def train_model(model, data_loader, epochs, criterion, optimizer, sum_writer, validation_loader,
                model_save_path, cuda_device=None):
    print('Starting training procedure.')
    min_valid_loss = np.inf
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
            _, preds = torch.max(outputs, 1)  # Pull the predictions.
            model_loss = criterion(outputs, labels)  # Compute training loss.

            model_loss.backward()  # Execute backpropagation.
            optimizer.step()  # Step in opposite direction to gradient.

            # Detach to avoid logging the entire gradient.
            ep_loss_run += model_loss.detach().item() * outputs.shape[
                0]  # Accumulate the epoch loss (avg with batch size).
            epoch_correct += (preds == labels).detach().sum().item()  # Accumulate number of correct class. each batch.
            total_samples += outputs.detach().size(0)  # Count number of samples seen so far.

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
            # Detach and remove training and validation datasets from memory.
            images.detach()
            del images
            labels.detach()
            del labels

        # Report everything at the end of each epoch: **************************************************************
        epoch_loss = ep_loss_run / total_samples
        epoch_acc = epoch_correct / total_samples

        # Write end of epoch results & Log this to tensorboard:
        timestamp = datetime.timestamp(datetime.now())
        print("Current Date & Time: ", timestamp, '\n')
        print(f'Epoch [{epoch + 1}/{epochs}], Epoch Training Loss: {epoch_loss:.4f},'
              f'Epoch Training Accuracy: {epoch_acc:.4f} \n')
        sum_writer.add_scalar('epoch_trn_loss', epoch_loss, epoch + 1)
        sum_writer.add_scalar('epoch_trn_acc', epoch_acc, epoch + 1)
        # **********************************************************************************************************

        # Try validation set at the end of each epoch:
        validate_model(model, validation_loader, epochs, criterion, optimizer, min_valid_loss, sum_writer,
                       model_save_path, cuda_device)


def validate_model(model, data_loader, epochs, criterion, optimizer, min_valid_loss, sum_writer,
                   model_save_path, cuda_device=None):
    print('Starting validation procedure.')
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
            _, preds = torch.max(outputs, 1)  # Pull the predictions.
            model_loss = criterion(outputs, labels)  # Compute training loss.

            ep_loss_run += model_loss.detach().item() * outputs.shape[
                0]  # Accumulate the epoch loss (avg with batch size).
            epoch_correct += (preds == labels).detach().sum().item()  # Accumulate number of correct class. each batch.
            total_samples += outputs.detach().size(0)  # Count number of samples seen so far.

            # Training Statistics & Logging: ***********************************************************************
            # At each batch record the data (both in terminal and tensorboard):
            if (idx + 1) % outputs.shape[0] == 0:
                timestamp = datetime.timestamp(datetime.now())
                print("Current Date & Time: ", timestamp, '\n')
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Step: [{idx + 1}/{len(data_loader)}], '
                      f'Validation Loss: {model_loss.item():.4f}, '
                      f'Validation Accuracy: {(epoch_correct / total_samples):.4f} \n'
                      )
                sum_writer.add_scalar('val_loss', ep_loss_run / total_samples,
                                      epoch * outputs.shape[0] + idx + 1)
                sum_writer.add_scalar('val_acc', (epoch_correct / total_samples),
                                      epoch * outputs.shape[0] + idx + 1)
            # ******************************************************************************************************
            # Detach and remove training and validation datasets from memory.
            images.detach()
            del images
            labels.detach()
            del labels

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


def test_model(model, test_loader, criterion, optimizer, sum_writer, cuda_device=None):
    print('Starting testing procedure.')
    # Assume that the best model is already loaded.
    model.eval()
    # Iterate over all batches of the training dataset:
    ep_loss_run = 0.0
    epoch_correct = 0
    total_samples = 0
    lbls_list = []
    prds_list = []
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

        # Store the labels and the predictions:
        lbls_list.append(labels.detach().cpu().numpy())
        prds_list.append(preds.detach().cpu().numpy())

        ep_loss_run += model_loss.detach().item() * outputs.shape[0]  # Accumulate the epoch loss (avg with batch size).
        epoch_correct += (preds == labels).detach().sum().item()  # Accumulate number of correct class. each batch.
        total_samples += outputs.detach().size(0)  # Count number of samples seen so far.

        # Detach and remove training and validation datasets from memory.
        images.detach()
        del images
        labels.detach()
        del labels

    # Report everything after passing through all the batches:
    epoch_loss = ep_loss_run / total_samples
    epoch_acc = epoch_correct / total_samples

    # Write end of epoch results & Log this to tensorboard:
    print(f'Test Dataset Loss: {epoch_loss:.4f},'
          f'Test Dataset Accuracy: {epoch_acc:.4f}')
    sum_writer.add_text('test_loss', f'Test Dataset Loss: {epoch_loss:.4f}', 0)
    sum_writer.add_text('test_accuracy', f'Test Dataset Accuracy: {epoch_acc:.4f}', 0)
    # **********************************************************************************************************
    return lbls_list, prds_list


# Showing closed figure: (IPYNB Supporting Implementation - For Debugging.)
# Taken from: https://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed
def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


if __name__ == '__main__':
    # Model Variables: (All used dataset and model samples listed here.)
    DATASET = ['mnist', 'cifar100']
    MODELS = ['vgg11', 'vgg11_bn', 'vgg_19', 'vgg19_bn', 'resnet18', 'resnet152', 'googlenet']

    # classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')  # for mnist
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # for cifar

    # Model training parameters: ***************************************************************************************
    EPOCHS = 1
    BATCH_SIZE = 4
    LR = 0.0001
    MOMENTUM = 0.9
    NUM_CLASSES = 10
    # ******************************************************************************************************************

    # Select the dataset and the model to train: ***********************************************************************
    selected_dataset = DATASET[0]
    selected_model = MODELS[3]
    # ******************************************************************************************************************

    # Directory Variables: (Created using the selected parameters above)
    data_dir = os.path.join('datasets', selected_dataset)
    trn_dir = os.path.join(data_dir, 'train')  # created automatically
    tst_dir = os.path.join(data_dir, 'test')  # created automatically

    save_dir = os.path.join('reporting', selected_dataset, selected_model)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_save_dir = os.path.join(save_dir, selected_model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Best model's save path.
    model_save = os.path.join(save_dir, selected_model + '.pth')

    # Create tensorboard writer:
    writer = SummaryWriter(os.path.join(save_dir, os.path.join('runs', selected_model)))

    # Log Timestamp of start, batch size, learning rate, epochs, momentum:
    timestamp = datetime.timestamp(datetime.now())
    print("Current Date & Time: ", timestamp, '\n')
    writer.add_text('start_time', f'Code started on  {timestamp}.', 0)
    writer.add_text('number_of_epochs', f'Total of {str(EPOCHS)} will be trained', 0)
    writer.add_text('batch_size', f'Batch size is {str(BATCH_SIZE)}.', 0)
    writer.add_text('learning_rate', f'Learning rate is {str(LR)}.', 0)
    writer.add_text('batch_size', f'Momentum is {str(MOMENTUM)}.', 0)
    writer.add_text('sel_dataset', f'Selected Dataset is {selected_dataset}.', 0)
    writer.add_text('sel_model', f'Selected Model is {selected_model}.', 0)

    # Create the device which the training will take place:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the dataset:
    tr_loader, vl_loader, tst_loader, _, _, _ = load_dataset(selected_dataset, 224, trn_dir, tst_dir, BATCH_SIZE, 3)
    demo_loader, _, _, _, _, _ = load_dataset(selected_dataset, 224, trn_dir, tst_dir, 16, 3)

    # Show some example input image-label pairs:
    input_fig = show_images(demo_loader)
    writer.add_figure('input_images', input_fig, 0)
    del demo_loader  # Remove memory

    # Create the model:
    my_model = load_model_template(selected_model, NUM_CLASSES, device)

    # Model hyperparameters:
    loss_criterion = nn.CrossEntropyLoss()  # create the loss criterion and the optimizer
    model_optm = torch.optim.SGD(my_model.parameters(), lr=LR, momentum=MOMENTUM)

    ims, _ = next(iter(tr_loader))  # Provide temporary images.
    writer.add_graph(my_model, ims.to(device))  # Add the model to tensorboard.
    del ims

    # Train the model:
    start = time.time()
    train_model(my_model, tr_loader, EPOCHS, loss_criterion, model_optm, writer, vl_loader,
                model_save, device)
    end = time.time()
    elapsed = end - start

    writer.add_text('training_elapsed', f'Training is done in {str(elapsed)} seconds', 0)

    # Reload the best model:
    my_model.load_state_dict(torch.load(model_save))

    # Test the model:
    start = time.time()
    labels_list, preds_list = test_model(my_model, tst_loader, loss_criterion, model_optm, writer, device)
    end = time.time()
    elapsed = end - start

    writer.add_text('inference_time', f'Testing is done in {str(elapsed)} seconds', 0)

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)

    # Outputs the same data loader as we down shuffle:
    _, _, _, _, _, tst = load_dataset(selected_dataset, 224, trn_dir, tst_dir, 1, 3)

    # Correct an False indices boolean array: (already ordered)
    # Pick 8 random choices from correct and incorrect samples:
    mask = labels_list == preds_list
    idx_corr = np.asarray(np.where(mask))[0]
    corr_picked = np.random.choice(idx_corr, 8)

    idx_false = np.asarray(np.where(~mask))[0]
    false_picked = np.random.choice(idx_false, 8)

    # Associate images with the indices and print them in a grid:
    o_fig = show_model_examples(tst.data.numpy(), idx_corr, idx_false, labels_list, preds_list)

    # Save in tensorboard:
    show_figure(o_fig)
    o_fig.show()
    writer.add_figure('output_images', o_fig, 0)

    # Confusion Matrix:
    cf_matrix = confusion_matrix(labels_list, preds_list)

    # print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, 0), index=[i for i in classes],
                         columns=[i for i in classes])
    cm_fig = plt.figure(figsize=(12, 7))
    cm_fig.suptitle('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, cmap='flare')
    writer.add_figure('test_confusion', cm_fig, 0)

    show_figure(cm_fig)
    cm_fig.show()

    writer.close()
