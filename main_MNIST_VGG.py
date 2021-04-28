# **********************************************************************************************************************
# QMUL - ECS 795P - Final Project - April 2021 - 200377106
# VGG Training using PyTorch on MNIST Dataset.
# Constructs a VGG model using PyTorch's implementation without any pretraining.
# *********************************************************************************************************************
import itertools
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from six.moves import urllib

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


# From: https://stackoverflow.com/questions/58589349/pytorch-confusion-matrix-plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for a, b in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(b, a, format(cm[a, b], fmt),
                 horizontalalignment="center",
                 color="white" if cm[a, b] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
    batch_size = 4

    # Model Parameters:
    image_scale = 224  # VGG accepts images of this size. (Resize input to this scale)

    # Load the training and testing datasets: (Apply normalization and build to tensors as well)
    # Added to circumvent Cloudflare protection bug, following from: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Normalization values of dataset from : https://discuss.pytorch.org/t/normalization-in-the-mnist-example/
    # Additional transforms can be added for data augmentation; Such as rotation:
    trnsfrm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.RandomRotation(15),
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
    vgg = [models.vgg11(pretrained=False),
           # models.vgg11_bn(pretrained=False),
           # models.vgg13(pretrained=False),
           # models.vgg13_bn(pretrained=False),
           # models.vgg16(pretrained=False),
           # models.vgg16_bn(pretrained=False),
           # models.vgg19(pretrained=False),
           models.vgg19_bn(pretrained=False),
           ]

    vgg_names = ["vgg11",
                 # "vgg11_bn",
                 # "vgg13",
                 # "vgg13_bn",
                 # "vgg16",
                 # "vgg16_bn",
                 # "vgg19",
                 "vgg19_bn"
                 ]

    # Loop over all possible models listed.
    for i in range(len(vgg)):
        # Training Parameters:
        learning_rate = 0.0001
        momentum = 0.9
        epochs = 1

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
        model = vgg[i]  # pick the model to work on
        model.classifier[6] = nn.Linear(4096, 10)  # set to number of classes in mnist
        model.to(device)  # send to cuda
        writer.add_graph(model, images)  # add the model to tensorboard

        loss_criterion = nn.CrossEntropyLoss()  # create the loss criterion and the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        print('Start training of: ', vgg_names[i], '.')
        for epoch in range(epochs):
            train_loss = 0.0
            train_corr = 0
            total = 0
            epoch_loss = 0.0
            epoch_acc = 0.0
            start_time = time.time()
            model.train()
            for k, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # forward pass
                _, preds = torch.max(outputs, 1)  # pull the predictions
                loss = loss_criterion(outputs, labels)  # compute training loss

                optimizer.zero_grad()  # clean the gradient
                loss.backward()  # backpropagation
                optimizer.step()  # step in opposite direction to gradient

                # Training Statistics & Logging:
                train_loss += loss.item() * images.size(0)  # Accumulate running loss.
                total += labels.size(0)
                train_corr += (preds == labels).sum().item()  # Accumulate running accuracy.

                # At each 100's batch record the data (both in terminal and tensorboard): ******************************
                if (k + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1} / {epochs}], '
                          f'Step: [{k + 1}/{len(train_loader)}], '
                          f'Training Loss: {(train_loss / total):.4f}, '
                          f'Training Accuracy: {(train_corr / total):.4f}'
                          )
                    writer.add_scalar('trn_loss', train_loss / (k + 1),
                                      epoch * outputs.shape[0] + k + 1)
                    writer.add_scalar('trn_acc', train_corr / (k + 1),
                                      epoch * outputs.shape[0] + k + 1)
                # ******************************************************************************************************
                # Report everything at the end of each epoch:
                epoch_loss = train_loss / total
                epoch_acc = train_corr / total

            # **********************************************************************************************************
            # Write end of epoch results & Log this to tensorboard:
            print(f'Epoch [{epoch + 1}/{epochs}], Mean Training Loss: {epoch_loss:.4f},'
                  f'Mean Training Accuracy: {epoch_acc:.4f}')
            writer.add_scalar('epoch_trn_loss', epoch_loss, epoch + 1)
            writer.add_scalar('epoch_trn_acc', epoch_acc, epoch + 1)
            # **********************************************************************************************************

            # ToDo: Use validation loss to achieve early stopping; LRScheduling etc.
            val_loss = 0.0
            val_corr = 0
            total = 0
            val_epoch_loss = 0.0
            val_epoch_acc = 0.0
            start_time = time.time()
            model.eval()
            min_valid_loss = np.inf
            for j, (vimages, vlabels) in enumerate(val_loader):
                vimages, vlabels = vimages.to(device), vlabels.to(device)
                with torch.no_grad():
                    voutputs = model(vimages)  # forward pass

                # Calculate and accumulate correct outputs: & Log them in tensorboard:
                _, predicted = torch.max(voutputs.data, 1)
                val_corr += (predicted == vlabels).sum().item()

                vloss = loss_criterion(voutputs, vlabels)  # compute loss
                optimizer.zero_grad()  # clean the gradient

                # Validation Statistics & Logging:
                val_loss += vloss.item() * vimages.size(0)  # Accumulate running loss.
                total += labels.size(0)
                val_corr += (predicted == vlabels).sum().item()  # Accumulate running accuracy.

                # ******************************************************************************************************
                # Report everything at the end of each epoch:
                val_epoch_loss = val_loss / total
                val_epoch_acc = val_corr / total

                # Save best models to drive:
                if min_valid_loss > val_epoch_loss:
                    print(
                        f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_epoch_loss:.6f}) \t Saving The Model')
                    min_valid_loss = val_epoch_loss

                # Saving State Dict:
                model_save_path = os.path.join(save_dir, str(vgg_names[i]))
                torch.save(model.state_dict(), model_save_path)

            # **********************************************************************************************************
            # Write end of epoch results & Log this to tensorboard:
            print(f'Epoch [{epoch + 1}/{epochs}], Mean Validation Loss: {val_epoch_loss:.4f},'
                  f'Mean Validation Accuracy: {val_epoch_acc:.4f}')
            writer.add_scalar('epoch_val_loss', val_epoch_loss, epoch + 1)
            writer.add_scalar('epoch_val_acc', val_epoch_acc, epoch + 1)
            # **********************************************************************************************************

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Training + Validation Time Elapsed: ', str(elapsed_time), 'seconds.')
            writer.add_scalar('epoch_time_elapsed', elapsed_time, epoch)

        # Reload the best model to start testing:
        model.load_state_dict(torch.load(model_save_path))  # model already created by now via the pipeline.
        model.eval()

        # Testing Procedure:
        correct = 0
        total = 0

        start_time = time.time()
        test_pred_labels = []
        test_actual_labels = []
        for j, (vimages, vlabels) in enumerate(test_loader):
            vimages, vlabels = vimages.to(device), vlabels.to(device)
            with torch.no_grad():
                voutputs = model(vimages)  # forward pass
            _, predicted = torch.max(voutputs.data, 1)

            temp = predicted.detach().cpu().numpy()  # convert to CPU then to numpy
            test_pred_labels.append(temp)  # append to the predicted labels
            test_actual_labels.append(vlabels.detach().cpu().numpy())

            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()
        end_time = time.time()

        acc = correct / total
        # Upon completion report final testing accuracy and inference speed:
        print(f'Testing Accuracy: {acc:.4f}')
        writer.add_text(str(vgg_names[i]), f'Testing Accuracy: {acc:.4f}')

        elapsed_time = end_time - start_time
        print('Inference Time of the Model: ', str(elapsed_time / len(test)), 's.')
        writer.add_text(str(vgg_names[i]), f'Inference Time of the Model: {str(elapsed_time / len(test))} s.')

        # Size of the model:
        total_params = sum(p.numel() for p in model.parameters())
        writer.add_text(str(vgg_names[i]), f'Total number of parameters: {str(total_params)}.')

        # Example Testing Image Pairings: ******************************************************************************
        test_im, test_lbl = next(iter(test_loader))
        test_im, test_lbl = test_im.to(device), test_lbl.to(device)

        # Construct the images: - Print as figure:
        # Create the subplot:
        fig, axs = plt.subplots(4, 4)
        fig.suptitle("Testing Images")
        for (ax, image, label) in zip(axs.flat, test_im[0:25], test_lbl[0:25]):
            ax.imshow(image[0].cpu(), cmap='gray')
            ax.set_title(f'Actual Label: {label}, -- Predicted Label {test_actual_labels}').set_fontsize('6')
            ax.axis('off')  # hide axes

        # Log images to tensorboard:
        writer.add_figure('output_images', fig, 0)
        # **************************************************************************************************************

        # ToDo:  Confusion Matrix for the best model:
        # Graphical analytics
        cm = confusion_matrix(test_actual_labels, test_pred_labels)
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(cm, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        # plot_confusion_matrix(cm, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        # Close the writer as we are done, new writer to be created next.
        writer.close()

        # Delete model to continue with the next model on GPU:
        del model
