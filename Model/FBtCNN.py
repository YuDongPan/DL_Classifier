# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/1 17:48
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SamePadConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Calculate the padding to achieve 'same' padding
        padding_length = ((kernel_size[1] // stride[1]) - 1) * stride[1]
        padding_left = padding_length // 2
        padding_right = padding_length - padding_left
        self.pad = nn.ZeroPad2d((padding_left, padding_right, 0, 0))  # left, right, top, bottom

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class tCNN(nn.Module):

    def filter_bank(self, X):
        low_lst = [3, 9, 14, 19]
        high_lst = [14, 26, 38, 50]

        device = X.device
        X = X.cpu().data.numpy()

        filter_X_lst = []
        for i in range(len(low_lst)):
            b, a = signal.butter(6, Wn=[2  * low_lst[0] / self.Fs, 2 * high_lst[0] / self.Fs], btype='bandpass')
            filter_X = signal.filtfilt(b, a, X, axis=-1)
            filter_X = torch.from_numpy(filter_X.copy()).float()
            filter_X = filter_X.to(device)
            filter_X_lst.append(filter_X)

        return filter_X_lst


    def spatial_block(self, in_channels, out_channels, kernel_size, stride, padding_mode):
        net = []
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                      stride=(stride, 1), padding=padding_mode))
        net.append(nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001))
        net.append(nn.ELU(alpha=1))
        net.append(nn.Dropout(0.4))
        net = nn.Sequential(*net)
        return net

    def temporal_block(self, in_channels, out_channels, kernel_size, stride, padding_mode):
        net = []
        if padding_mode == "same":
            net.append(SamePadConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                     stride=(1, stride)))
        else:
            net.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                 stride=(1, stride), padding=padding_mode))

        net.append(nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001))
        net.append(nn.ELU(alpha=1))
        net.append(nn.Dropout(0.4))
        net = nn.Sequential(*net)
        return net

    def __init__(self, Nc, Nt, Nf, Fs):
        super(tCNN, self).__init__()
        self.Nc = Nc
        self.Nt = Nt
        self.Nf = Nf
        self.Fs = Fs
        self.K = self.Nt // 5 - 4

        # First Convolution Layer (Data Format: 1 @ Nc × Nt)
        self.conv_block1 = self.spatial_block(in_channels=1, out_channels=16, kernel_size=self.Nc, stride=self.Nc,
                                              padding_mode="valid")

        # Second Convolution Layer (Data Format: 16 @ 1 × Nt)
        self.conv_block2 = self.temporal_block(in_channels=16, out_channels=16, kernel_size=self.Nt, stride=5,
                                               padding_mode="same")

        # Third Convolution Layer (Data Format: 16 @ 1 × Nt // 5)
        self.conv_block3 = self.temporal_block(in_channels=16, out_channels=16, kernel_size=5, stride=1,
                                               padding_mode="valid")

        # Sub-band Networks (Data Format: 16 @ 1 × (Nt // 5 - 4))
        self.sub_band = nn.Sequential(self.conv_block1, self.conv_block2, self.conv_block3)

        # Fourth Convolution Layer (Data Format: 16 @ 1 × (Nt // 5 - 4))
        self.conv_block4 = self.temporal_block(in_channels=16, out_channels=32, kernel_size=self.K, stride=1,
                                               padding_mode="valid")

        # Dense Layer (Data Format: 32 @ 1 × 1)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=self.Nf))

        # Output: (bz, Nf)


    def forward(self, x):
        # Filter banks generation
        filter_x = self.filter_bank(x)
        x1, x2, x3, x4 = filter_x[0], filter_x[1], filter_x[2], filter_x[3]

        # Sub-band Parallel Input: Conv1-Conv3 blocks
        x1 = self.sub_band(x1)
        x2 = self.sub_band(x2)
        x3 = self.sub_band(x3)
        x4 = self.sub_band(x4)

        # Sub-band features fusion
        x = x1 + x2 + x3 + x4

        # Conv4 block
        x = self.conv_block4(x)

        # Dense Layer
        out = self.dense(x)

        return out