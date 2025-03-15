# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - CycleGenerator     --> Used in the CycleGAN in Part 2
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
#   - PatchDiscriminator --> Used in the CycleGAN in Part 2
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, CycleGenerator, DCDiscriminator, and PatchDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.up_conv1 = conv(noise_size, conv_dim*8, 4, 1, 3, 'instance', True, 'relu')
        self.up_conv2 = up_conv(conv_dim*8, conv_dim*4, 3, 1, 1, activ='relu')
        self.up_conv3 = up_conv(conv_dim*4, conv_dim*2, 3, 1, 1, activ='relu')
        self.up_conv4 = up_conv(conv_dim*2, conv_dim, 3, 1, 1, activ='relu')
        self.up_conv5 = up_conv(conv_dim, 3, 3, 1, 1, activ='tanh')

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        
        # Pass the input through each layer in sequence
        out = self.up_conv1(z)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        
        return out.squeeze()


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Architecture of the generator network."""

    def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance'):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4, 2, 1, 'instance', init_zero_weights, 'relu')
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, 'instance', False, 'relu')

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim*2, 'instance', 'relu')

        # 3. Define the decoder part of the generator
        self.up_conv1 = up_conv(conv_dim*2, conv_dim, 3, 1, 1, norm='instance', activ='relu')
        self.up_conv2 = up_conv(conv_dim, 3, 3, 1, 1, norm='None', activ='tanh')

    def forward(self, x):
        """
        Generate an image conditioned on an input image.

        Input
        -----
            x: BS x 3 x 32 x 32

        Output
        ------
            out: BS x 3 x 32 x 32
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.resnet_block(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)

        return out.squeeze()


class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm='batch'):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu')
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, norm, False, 'relu')
        self.conv3 = conv(conv_dim*2, conv_dim * 4, 4, 2, 1, norm, False, 'relu')
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 2, 1, norm, False, 'relu')
        self.conv5 = conv(conv_dim * 8, 1, 4, 2, 0, None, False, None)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()

class PatchDiscriminator(nn.Module):
    """Architecture of the patch discriminator network."""

    def __init__(self, conv_dim=64, norm='batch'):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # Hint: it should look really similar to DCDiscriminator.
        self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu')
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, norm, False, 'relu')
        self.conv3 = conv(conv_dim*2, conv_dim * 4, 4, 2, 1, norm, False, 'relu')
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 2, 1, norm, False, 'relu')
        self.conv5 = conv(conv_dim * 8, 1, 4, 2, 0, None, False, None)




    def forward(self, x):

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


if __name__ == "__main__":
    a = torch.rand(4, 3, 64, 64)
    D = PatchDiscriminator()
    print(D(a).shape)
    G = CycleGenerator()
    print(G(a).shape)
