# import os
# import time

import torch.nn as nn
# from tqdm import tqdm  # Time Counter:


class Resnet(nn.Module):
    """ Baseline implementation of ResNet architecture for image classification.
    """

    def __init__(self, input_dims, output_dims):
        """Initialize the network structure as described in ResNet paper.

        Arguments: Todo
            input_dims {int} -- Dimension of input images
            output_dims {int} -- Dimension of the output
        """
        super().__init__()
        # super(Generator, self).__init__()
        # ###  Changed the architecture and value as CW2 Guidance required ###
        # self.model = nn.Sequential(nn.Linear(input_dims, 256), nn.LeakyReLU(0.2),
        #                            nn.Linear(256, 512), nn.LeakyReLU(0.2),
        #                            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
        #                            nn.Linear(1024, output_dims), nn.Tanh()
        #                            )

    # def forward(self, x):
    #     """Forward function
    #
    #     Arguments:
    #         x {Tensor} -- a batch of noise vectors in shape (<batch_size>x<input_dims>)
    #
    #     Returns:
    #         Tensor -- a batch of flatten image in shape (<batch_size>x<output_dims>)
    #     """
    #     ###  Modified to be consistent with the network structure. ###
    #     return self.model(x)
