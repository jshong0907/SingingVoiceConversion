import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DownSampleBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), bias=False)

    def forward(self, x):
        x = nn.InstanceNorm2d(x.shape[1])(self.conv(x))
        x = nn.GLU(dim=1)(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=(5, 5),
                               stride=(1, 1), padding=(2, 2), bias=False)

    def forward(self, x):
        x = nn.PixelShuffle(2)(nn.InstanceNorm2d(x.shape[1])(self.conv(x)))
        x = nn.GLU(dim=1)(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3),
                               stride=(1, 1), padding=(0, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 3),
                               stride=(1, 1), padding=(0, 1), bias=False)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = nn.InstanceNorm2d(x.shape[1])(x)
        x = nn.GLU(dim=1)(x)
        x = self.conv2(x)
        x = nn.InstanceNorm2d(x.shape[1])(x)
        return x + res


class Generator(nn.Module):
    def __init__(self, num_features):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15),
                               stride=(1, 1), padding=(2, 7), bias=False)

        self.downsample_block1 = DownSampleBlock(inchannels=64, outchannels=256)
        self.downsample_block2 = DownSampleBlock(inchannels=128, outchannels=512)

        self.conv2 = nn.Conv2d(in_channels=int(int((num_features+1)/2 + 1)/2)*256, out_channels=256, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)

        for i in range(6):
            self.add_module("residual_block" + str(i+1), ResidualBlock())

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=int(int((num_features+1)/2 + 1)/2)*256, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)

        self.upsample_block1 = UpSampleBlock(inchannels=256, outchannels=1024)
        self.upsample_block2 = UpSampleBlock(inchannels=128, outchannels=512)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=num_features, kernel_size=(5, 15),
                               stride=(1, 1), padding=(2, 7), bias=False)

        self.output = nn.Conv2d(in_channels=int(int((num_features+1)/2 + 1)/2)*4, out_channels=1, kernel_size=(5, 15),
                                stride=(1, 1), padding=(2, 7), bias=False)


    def forward(self, x):
        time = x.shape[2]
        num_featuers = x.shape[3]
        x = self.conv1(x)
        x = nn.GLU(dim=1)(x)
        for i in range(2):
            x = self.__getattr__("downsample_block" + str(i+1))(x)
        x = x.reshape((x.shape[0], -1, int(int((time+1)/2 + 1)/2), 1))
        x = self.conv2(x)
        x = nn.InstanceNorm2d(x.shape[1])(x)

        for i in range(6):
            x = self.__getattr__("residual_block" + str(i+1))(x)

        x = self.conv3(x)
        x = nn.InstanceNorm2d(x.shape[1])(x)
        x = x.reshape(x.shape[0], -1, int(int((time+1)/2 + 1)/2), int(int((num_featuers+1)/2 + 1)/2))

        for i in range(2):
            x = self.__getattr__("upsample_block" + str(i+1))(x)

        x = self.conv4(x)

        x = x.reshape(x.shape[0], x.shape[3], x.shape[2], x.shape[1])

        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self, width, height):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1), bias=False)

        self.downsample_block1 = DownSampleBlock(inchannels=64, outchannels=256)
        self.downsample_block2 = DownSampleBlock(inchannels=128, outchannels=512)
        self.downsample_block3 = DownSampleBlock(inchannels=256, outchannels=1024)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 5),
                                stride=(1, 1), padding=(0, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 3),
                                stride=(1, 1), padding=(0, 1), bias=False)

        self.fc = nn.Linear(int((width+7)/8) * int((height+7)/8), 1)

    def forward(self, x):
        time = x.shape[2]
        num_featuers = x.shape[3]
        x = self.conv1(x)
        x = nn.GLU(dim=1)(x)

        for i in range(3):
            x = self.__getattr__("downsample_block" + str(i+1))(x)

        x = self.conv2(x)
        x = nn.InstanceNorm2d(x.shape[1])(x)
        x = nn.GLU(dim=1)(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return nn.Sigmoid()(x)
