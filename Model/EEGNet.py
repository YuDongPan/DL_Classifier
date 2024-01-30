# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:05
import torch
from torch import nn
from Utils.Constraint import Conv2dWithConstraint

class EEGNet(nn.Module):
    def Depthwise_Separable_Conv(self, X, depthwiseConv, pointwiseConv):
        X = depthwiseConv(X)
        X = pointwiseConv(X)
        return X

    def __init__(self, num_channels, T, num_classes):
        '''Constructing Function'''
        super(EEGNet, self).__init__()
        self.F1 = 96
        self.D = 1
        self.F2 = self.F1 * self.D
        self.kernelength1 = T
        self.kernelength2 = T // 16
        self.p = 0.5
        self.fc_in = self.F2 * (T // 32)

        # Define Activation Function and Dropout
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(self.p)

        # First Layer:Temporal Convolution + Spatial Convolution
        '''时间卷积'''
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernelength1),  bias=False, padding="same")
        self.bn1_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        '''空间卷积'''
        self.Depthwise_Convolution = Conv2dWithConstraint(self.F1, self.F1 * self.D, (num_channels, 1), max_norm=1,
                                                          groups=self.F1, bias=False)

        self.bn1_2 = nn.BatchNorm2d(self.D * self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Second Layer:Separable Convolution
        self.depthwiseConv2 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelength2), groups=self.F1 * self.D,
                                        bias=False, padding="same")
        self.pointwiseConv2 = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # Third Layer:Fully Connected + Softmax
        self.fc = nn.Linear(in_features=self.fc_in, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, X):
        '''前向计算'''
        # First Layer:Temporal Convolution + Spatial Convolution
        X = self.conv1(X)
        X = self.bn1_1(X)
        X = self.Depthwise_Convolution(X)
        X = self.bn1_2(X)
        X = self.activation(X)
        X = self.pool1(X)
        X = self.dropout(X)

        # Second Layer:Separable Convolution
        X = self.Depthwise_Separable_Conv(X, self.depthwiseConv2, self.pointwiseConv2)
        X = self.bn2(X)
        X = self.activation(X)
        X = self.pool2(X)
        X = self.dropout(X)

        # Third Layer:Fully Connected + Softmax
        X = X.reshape(-1, self.fc_in)
        out = self.fc(X)
        return out